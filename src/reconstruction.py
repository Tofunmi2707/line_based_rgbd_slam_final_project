from pathlib import Path
import open3d as o3d

from src.tum_io import build_depth_index, nearest_depth_path


def fuse_rgbd_from_poses(dataset_cfg,
                         image_files,
                         poses_wc,
                         output_dir: Path,
                         max_frames=60,
                         step=2,
                         visualise=True):
    output_dir.mkdir(parents=True, exist_ok=True)

    fx = dataset_cfg.intrinsics[0, 0]
    fy = dataset_cfg.intrinsics[1, 1]
    cx = dataset_cfg.intrinsics[0, 2]
    cy = dataset_cfg.intrinsics[1, 2]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480, fx=fx, fy=fy, cx=cx, cy=cy
    )

    depth_ts, depth_files = build_depth_index(dataset_cfg.depth_dir)

    fused = o3d.geometry.PointCloud()
    used_frames = 0

    for idx in range(0, min(max_frames, len(image_files)), step):
        rgb_path = Path(image_files[idx])
        depth_path = nearest_depth_path(
            rgb_path, depth_ts, depth_files, dataset_cfg.max_rgb_depth_dt
        )

        if depth_path is None:
            continue

        color = o3d.io.read_image(str(rgb_path))
        depth = o3d.io.read_image(str(depth_path))

        rgbd = o3d.geometry.RGBDImage.create_from_tum_format(
            color, depth, convert_rgb_to_intensity=False
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
        fused, _ = fused.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)

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
