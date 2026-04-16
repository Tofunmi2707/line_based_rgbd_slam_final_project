from __future__ import annotations

"""
Generate a rotating GIF from a fused point-cloud reconstruction.

This script renders a saved point cloud from a fixed initial viewpoint and
captures an orbiting sequence of frames, which are then exported as an animated
GIF. The output is intended as a supplementary qualitative artefact for the
dissertation and appendix.

Inspiration:
- Open3D visualiser and view-control APIs for point-cloud rendering.
- The orbit-rendering sequence and GIF export workflow were integrated within
  the present project for examiner-friendly visualisation of reconstructed
  scenes.
"""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_rotating_gif(
    ply_path: Path,
    gif_path: Path,
    frame_dir: Path,
    width: int = 1280,
    height: int = 960,
    num_frames: int = 72,
    point_size: float = 2.0,
    front: tuple[float, float, float] = (0.4, -0.5, -0.75),
    up: tuple[float, float, float] = (0.0, -1.0, 0.0),
    zoom: float = 0.55,
) -> None:
    """
    Render a rotating GIF from a saved point cloud.

    The function:
    1. loads the point cloud,
    2. opens an off-screen Open3D visualiser,
    3. sets a fixed initial viewpoint,
    4. rotates the camera incrementally around the scene,
    5. saves each rendered frame,
    6. combines the frames into an animated GIF.

    Args:
        ply_path: Path to the input point-cloud file.
        gif_path: Path to the output GIF file.
        frame_dir: Directory in which the
                   intermediate rendered frames are saved.
        width: Render width in pixels.
        height: Render height in pixels.
        num_frames: Number of frames in the orbit sequence.
        point_size: Rendered point size.
        front: Initial camera front vector.
        up: Initial camera up vector.
        zoom: Initial camera zoom level.

    Returns:
        None

    Raises:
        RuntimeError: If the loaded point cloud is empty.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"Point cloud is empty: {ply_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pcd)

    render = vis.get_render_option()
    render.background_color = np.array([1.0, 1.0, 1.0])
    render.point_size = point_size
    render.light_on = True

    ctr = vis.get_view_control()
    bbox = pcd.get_axis_aligned_bounding_box()
    lookat = bbox.get_center().tolist()

    ctr.set_front(list(front))
    ctr.set_lookat(lookat)
    ctr.set_up(list(up))
    ctr.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()

    images = []
    for i in range(num_frames):
        ctr.rotate(10.0, 0.0)
        vis.poll_events()
        vis.update_renderer()

        frame_path = frame_dir / f"frame_{i:03d}.png"
        vis.capture_screen_image(str(frame_path), do_render=True)
        images.append(imageio.imread(frame_path))

    vis.destroy_window()
    imageio.mimsave(str(gif_path), images, duration=0.08)


def main() -> None:
    """
    Generate the rotating GIF for the selected reported reconstruction.

    Returns:
        None
    """
    dataset_name = "fr1_desk"
    method_name = "v3_geom_filter"

    ply_path = PROJECT_ROOT / "results" / dataset_name / method_name / "fused_cloud_3d.ply"  # noqa: E501
    gif_path = PROJECT_ROOT / "results" / "gifs" / f"{dataset_name}_{method_name}_rotation.gif"  # noqa: E501
    frame_dir = PROJECT_ROOT / "results" / "gifs" / f"{dataset_name}_{method_name}_frames"  # noqa: E501

    make_rotating_gif(ply_path, gif_path, frame_dir)
    print(f"Saved GIF to: {gif_path}")


if __name__ == "__main__":
    main()
