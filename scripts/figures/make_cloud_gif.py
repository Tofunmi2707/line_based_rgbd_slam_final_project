from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import open3d as o3d

# Inspiration: Open3D visualiser and view-control API; orbit rendering sequence
# and report-specific GIF export implemented for this project.

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def make_rotating_gif(
    ply_path: Path,
    gif_path: Path,
    frame_dir: Path,
    width: int = 1280,
    height: int = 960,
    num_frames: int = 72,
    point_size: float = 2.0,
    front = (0.4, -0.5, -0.75),
    up = (0.0, -1.0, 0.0),
    zoom: float = 0.55,
):
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
        ctr.rotate(10.0, 0.0)   # clockwise horizontal orbit
        vis.poll_events()
        vis.update_renderer()

        frame_path = frame_dir / f"frame_{i:03d}.png"
        vis.capture_screen_image(str(frame_path), do_render=True)
        images.append(imageio.imread(frame_path))

    vis.destroy_window()
    imageio.mimsave(str(gif_path), images, duration=0.08)

def main():
    dataset_name = "fr1_desk"
    method_name = "v3_geom_filter"

    ply_path = PROJECT_ROOT / "results" / dataset_name / method_name / "fused_cloud_3d.ply"
    gif_path = PROJECT_ROOT / "results" / "gifs" / f"{dataset_name}_{method_name}_rotation.gif"
    frame_dir = PROJECT_ROOT / "results" / "gifs" / f"{dataset_name}_{method_name}_frames"

    make_rotating_gif(ply_path, gif_path, frame_dir)
    print(f"Saved GIF to: {gif_path}")

if __name__ == "__main__":
    main()
