from __future__ import annotations

"""
Generate reconstruction comparison figures for the dissertation.

This script renders saved point clouds from a fixed viewpoint and composes
report-ready comparison figures. It is used to produce:
- rendered point-cloud images for baseline and final methods,
- a side-by-side baseline vs final reconstruction figure,
- a side-by-side scene vs fused-cloud figure.

Inspiration:
- Open3D hidden-window rendering through the Visualizer and ViewControl APIs.
- The figure layout and baseline/final comparison structure were integrated
  within the present project for the reconstruction analysis section.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS  # noqa: E402
from src.tum_io import load_rgb_sequence  # noqa: E402


def load_rgb_frame(dataset_name: str, frame_idx: int) -> np.ndarray:
    """
    Load one RGB frame from the selected dataset.

    Args:
        dataset_name: Dataset name defined in the project configuration.
        frame_idx: Frame index to load.

    Returns:
        Loaded RGB image in BGR format.

    Raises:
        IndexError: If the requested frame index is out of range.
        FileNotFoundError: If the frame cannot be loaded.
    """
    dataset_cfg = DATASETS[dataset_name]
    rgb_files = load_rgb_sequence(dataset_cfg.rgb_dir)

    if frame_idx < 0 or frame_idx >= len(rgb_files):
        raise IndexError(
            f"frame_idx {frame_idx} out of range for {dataset_name} "
            f"(0 to {len(rgb_files) - 1})"
        )

    img = cv2.imread(str(rgb_files[frame_idx]))
    if img is None:
        raise FileNotFoundError(f"Could not load RGB frame: "
                                f"{rgb_files[frame_idx]}")
    return img


def add_label(img_bgr: np.ndarray, label: str) -> np.ndarray:
    """
    Add a white label banner to the top of an image.

    Args:
        img_bgr: Input image in BGR format.
        label: Text label to draw.

    Returns:
        Labelled image.
    """
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 50), (255, 255, 255), -1)
    cv2.putText(
        out,
        label,
        (20, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def resize_to_same_height(
    imgs: list[np.ndarray],
    target_height: int,
) -> list[np.ndarray]:
    """
    Resize a list of images to a common height while preserving aspect ratio.

    Args:
        imgs: Input images.
        target_height: Desired output height.

    Returns:
        Resized images.
    """
    resized = []
    for img in imgs:
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(round(w * scale))
        resized.append(
            cv2.resize(
                img,
                (new_w, target_height),
                interpolation=cv2.INTER_AREA,
            )
        )
    return resized


def render_point_cloud_to_image(
    ply_path: Path,
    out_path: Path,
    width: int = 1400,
    height: int = 900,
    point_size: float = 2.0,
    front: list[float] | None = None,
    lookat: list[float] | None = None,
    up: list[float] | None = None,
    zoom: float | None = None,
) -> None:
    """
    Render a point cloud to an image using an off-screen Open3D visualiser.

    Args:
        ply_path: Path to the input point-cloud file.
        out_path: Output image path.
        width: Render width in pixels.
        height: Render height in pixels.
        point_size: Rendered point size.
        front: Camera front vector.
        lookat: Camera look-at point.
        up: Camera up vector.
        zoom: Camera zoom level.

    Returns:
        None

    Raises:
        FileNotFoundError: If the point-cloud file is missing.
        RuntimeError: If the point cloud contains no points.
    """
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing point cloud: {ply_path}")

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"Point cloud has no points: {ply_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Render {ply_path.stem}",
        width=width,
        height=height,
        visible=False,
    )
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([1.0, 1.0, 1.0])
    render_opt.point_size = point_size
    render_opt.light_on = True

    ctr = vis.get_view_control()
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    if front is None:
        front = [0.4, -0.5, -0.75]
    if lookat is None:
        lookat = center.tolist()
    if up is None:
        up = [0.0, -1.0, 0.0]
    if zoom is None:
        zoom = 0.55

    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()


def save_side_by_side(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_label: str,
    right_label: str,
    out_path: Path,
    height: int = 700,
) -> None:
    """
    Save a side-by-side comparison image with labels.

    Args:
        left_bgr: Left image in BGR format.
        right_bgr: Right image in BGR format.
        left_label: Label for the left image.
        right_label: Label for the right image.
        out_path: Output image path.
        height: Common output height.

    Returns:
        None
    """
    left_l = add_label(left_bgr, left_label)
    right_l = add_label(right_bgr, right_label)

    left_r, right_r = resize_to_same_height([left_l, right_l], height)
    canvas = np.hstack([left_r, right_r])
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    """
    Generate the main reconstruction comparison figures.

    The script renders baseline and final fused clouds from the same viewpoint,
    then saves:
    - baseline rendered cloud,
    - final rendered cloud,
    - baseline vs final comparison figure,
    - representative RGB scene vs final cloud figure.

    Returns:
        None
    """
    dataset_name = "fr1_desk"
    baseline_method = "v1_centroid"
    final_method = "v3_geom_filter"
    frame_idx = 100

    results_dir = PROJECT_ROOT / "results" / dataset_name
    out_dir = results_dir / "reconstruction_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_ply = results_dir / baseline_method / "fused_cloud_3d.ply"
    final_ply = results_dir / final_method / "fused_cloud_3d.ply"

    baseline_img_path = out_dir / f"{dataset_name}_{baseline_method}_cloud.png"
    final_img_path = out_dir / f"{dataset_name}_{final_method}_cloud.png"

    front = [0.4, -0.5, -0.75]
    up = [0.0, -1.0, 0.0]
    zoom = 0.55

    render_point_cloud_to_image(
        baseline_ply,
        baseline_img_path,
        front=front,
        up=up,
        zoom=zoom,
    )

    render_point_cloud_to_image(
        final_ply,
        final_img_path,
        front=front,
        up=up,
        zoom=zoom,
    )

    baseline_img = cv2.imread(str(baseline_img_path))
    final_img = cv2.imread(str(final_img_path))
    if baseline_img is None or final_img is None:
        raise RuntimeError("Failed to load rendered point-cloud images.")

    comparison_fig = out_dir / f"{dataset_name}_baseline_vs_final_cloud.png"
    save_side_by_side(
        baseline_img,
        final_img,
        f"Baseline: {baseline_method}",
        f"Final: {final_method}",
        comparison_fig,
    )

    rgb_frame = load_rgb_frame(dataset_name, frame_idx)
    scene_vs_cloud_path = out_dir / f"{dataset_name}_scene_vs_final_cloud.png"

    save_side_by_side(
        rgb_frame,
        final_img,
        f"Representative RGB frame ({dataset_name})",
        f"Fused point cloud: {final_method}",
        scene_vs_cloud_path,
    )

    print(f"Saved reconstruction figures to: {out_dir}")
    print(f"Baseline cloud image : {baseline_img_path}")
    print(f"Final cloud image    : {final_img_path}")
    print(f"Comparison figure    : {comparison_fig}")
    print(f"Scene vs cloud       : {scene_vs_cloud_path}")


if __name__ == "__main__":
    main()
