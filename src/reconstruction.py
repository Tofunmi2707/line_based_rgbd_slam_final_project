from __future__ import annotations

"""
RGB-D point-cloud reconstruction and multi-frame fusion utilities.

This module reconstructs coloured 3D point clouds from RGB-D frames using the
dataset camera intrinsics and estimated camera poses. It is used to generate
single fused scene reconstructions for qualitative evaluation of mapping
quality.

Inspiration:
- RGB-D image creation and point-cloud generation follow the standard Open3D
  workflow for depth-based reconstruction.
- RGB/depth pairing follows the project dataset-loading utilities for TUM RGB-D
  sequences.
- The fusion loop, output handling, and chosen filtering steps were integrated
  within the present project for reconstruction evaluation.
"""

from pathlib import Path
import open3d as o3d

from src.tum_io import build_depth_index, nearest_depth_path


def fuse_rgbd_from_poses(
    dataset_cfg,
    image_files,
    poses_wc,
    output_dir: Path,
    max_frames: int = 60,
    step: int = 2,
    visualise: bool = True,
) -> Path:
    """
    Fuse multiple RGB-D frames into a single coloured point cloud.

    The function:
    1. loads RGB frames and their nearest associated depth frames,
    2. constructs Open3D RGB-D images,
    3. back-projects each frame into a point cloud using the dataset
       intrinsics,
    4. transforms each cloud into the shared world frame using the
       estimated poses,
    5. accumulates the transformed clouds,
    6. applies voxel downsampling and statistical outlier removal,
    7. saves the fused point cloud to disk.

    Inspiration:
    - Open3D RGB-D reconstruction workflow using
      `RGBDImage.create_from_tum_format` and
      `PointCloud.create_from_rgbd_image`.
    - The frame-window fusion design is a project-specific evaluation tool used
      to study how pose quality affects reconstruction coherence.

    Args:
        dataset_cfg: Dataset configuration object containing intrinsics
        and paths.
        image_files: Sequence of RGB image paths used in the trajectory.
        poses_wc: Estimated world-frame camera poses aligned with the
        RGB frames.
        output_dir: Directory in which the fused point cloud should be saved.
        max_frames: Maximum number of frames to consider for fusion.
        step: Step size used when subsampling frames.
        visualise: Whether to display the fused cloud after saving.

    Returns:
        Path to the saved fused point-cloud file.

    Raises:
        RuntimeError: If no valid points are fused from the selected frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

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

    depth_ts, depth_files = build_depth_index(dataset_cfg.depth_dir)

    fused = o3d.geometry.PointCloud()
    used_frames = 0

    for idx in range(0, min(max_frames, len(image_files)), step):
        rgb_path = Path(image_files[idx])
        depth_path = nearest_depth_path(
            rgb_path,
            depth_ts,
            depth_files,
            dataset_cfg.max_rgb_depth_dt,
        )

        if depth_path is None:
            continue

        color = o3d.io.read_image(str(rgb_path))
        depth = o3d.io.read_image(str(depth_path))

        rgbd = o3d.geometry.RGBDImage.create_from_tum_format(
            color,
            depth,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        if len(pcd.points) == 0:
            continue

        pcd.transform(poses_wc[idx])
        fused += pcd
        used_frames += 1

    if len(fused.points) == 0:
        raise RuntimeError("No valid fused point cloud points.")

    fused = fused.voxel_down_sample(voxel_size=0.01)
    if len(fused.points) > 0:
        fused, _ = fused.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0,
        )

    out_path = output_dir / "fused_cloud_3d.ply"
    o3d.io.write_point_cloud(str(out_path), fused)

    print(f"Used frames for fusion: {used_frames}")
    print(f"Saved fused point cloud to {out_path}")

    if visualise:
        o3d.visualization.draw_geometries(
            [fused],
            window_name=f"Fused cloud - {dataset_cfg.name}",
            width=1200,
            height=800,
        )

    return out_path
