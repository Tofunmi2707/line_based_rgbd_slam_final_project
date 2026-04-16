from __future__ import annotations

# Inspiration: Open3D RGB-D image and point-cloud creation workflow using TUM
# conventions; project-specific file organisation, dataset loading, and report
# output handling implemented here.

from pathlib import Path
import sys

import cv2
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS  # noqa: E402
from src.tum_io import (  # noqa: E402
    load_rgb_sequence,
    build_depth_index,
    nearest_depth_path
)


def make_single_frame_cloud(
    dataset_name: str = "fr1_desk",
    frame_idx: int = 100,
    output_name: str = "single_frame_cloud.ply",
    visualise: bool = True,
) -> Path:
    # Inspiration: Open3D create_from_tum_format and create_from_rgbd_image;
    # wrapped here to generate the single-frame validation result used in the
    # reconstruction section of the dissertation.
    dataset_cfg = DATASETS[dataset_name]

    rgb_files = load_rgb_sequence(dataset_cfg.rgb_dir)
    if frame_idx < 0 or frame_idx >= len(rgb_files):
        raise IndexError(
            f"frame_idx {frame_idx} out of range for {dataset_name} "
            f"(0 to {len(rgb_files) - 1})"
        )

    rgb_path = rgb_files[frame_idx]

    depth_ts, depth_files = build_depth_index(dataset_cfg.depth_dir)
    depth_path = nearest_depth_path(
        rgb_path,
        depth_ts,
        depth_files,
        dataset_cfg.max_rgb_depth_dt,
    )

    if depth_path is None:
        raise RuntimeError("No matching depth frame found"
                           f" for {rgb_path.name}")

    color = o3d.io.read_image(str(rgb_path))
    depth = o3d.io.read_image(str(depth_path))

    rgbd = o3d.geometry.RGBDImage.create_from_tum_format(
        color,
        depth,
        convert_rgb_to_intensity=False,
    )

    fx = dataset_cfg.intrinsics[0, 0]
    fy = dataset_cfg.intrinsics[1, 1]
    cx = dataset_cfg.intrinsics[0, 2]
    cy = dataset_cfg.intrinsics[1, 2]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    if len(pcd.points) == 0:
        raise RuntimeError("Generated point cloud has no points.")

    out_dir = PROJECT_ROOT / "results" / dataset_name / "single_frame_cloud"
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_path = out_dir / output_name
    img_copy_path = out_dir / f"rgb_frame_{frame_idx}.png"

    o3d.io.write_point_cloud(str(ply_path), pcd)

    rgb_bgr = cv2.imread(str(rgb_path))
    if rgb_bgr is not None:
        cv2.imwrite(str(img_copy_path), rgb_bgr)

    print(f"Dataset          : {dataset_name}")
    print(f"Frame index      : {frame_idx}")
    print(f"RGB frame        : {rgb_path}")
    print(f"Depth frame      : {depth_path}")
    print(f"Saved point cloud: {ply_path}")
    print(f"Saved RGB copy   : {img_copy_path}")
    print(f"Point count      : {len(pcd.points)}")

    if visualise:
        window_title = "Single-frame cloud - "
        f"{dataset_name} - frame {frame_idx}"
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_title,
            width=1200,
            height=800,
        )

    return ply_path


def main() -> None:
    # Inspiration: dissertation single-frame reconstruction validation figure.
    make_single_frame_cloud(
        dataset_name="fr1_desk",
        frame_idx=100,
        output_name="single_frame_cloud.ply",
        visualise=True,
    )


if __name__ == "__main__":
    main()
