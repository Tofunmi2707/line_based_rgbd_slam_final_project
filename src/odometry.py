import numpy as np
from datetime import datetime

from src.tum_io import load_rgb_sequence, build_depth_index
from src.tum_io import nearest_depth_path, timestamp_from_path
from src.line_frontend_v2_lbd_endpoints import process_frame_pair_frontend
from src.pose_estimation import process_frame_pair_pose, make_T


def yaw_from_R(R: np.ndarray) -> float:
    return float(np.arctan2(R[0, 2], R[2, 2]))


def run_visual_odometry(dataset_cfg, odom_cfg):
    rgb_files = load_rgb_sequence(dataset_cfg.rgb_dir)
    depth_ts, depth_files = build_depth_index(dataset_cfg.depth_dir)

    odom_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    poses_wc = [np.eye(4, dtype=float)]
    timestamps = [timestamp_from_path(rgb_files[0])]
    odometry_edges = []
    debug_rows = []

    accepted_count = 0
    rejected_count = 0

    for i in range(len(rgb_files) - 1):
        f1 = rgb_files[i]
        f2 = rgb_files[i + 1]

        d1 = nearest_depth_path(
            f1, depth_ts, depth_files, dataset_cfg.max_rgb_depth_dt
        )
        d2 = nearest_depth_path(
            f2, depth_ts, depth_files, dataset_cfg.max_rgb_depth_dt
        )

        row = {
            "frame_i": i,
            "frame_j": i + 1,
            "timestamp_i": float(timestamp_from_path(f1)),
            "timestamp_j": float(timestamp_from_path(f2)),
            "raw_matches": 0,
            "filtered_matches": 0,
            "essential_inliers": 0,
            "metric_points": 0,
            "step_norm": 0.0,
            "accepted": False,
            "reject_reason": None,
        }

        front = process_frame_pair_frontend(str(f1), str(f2), odom_cfg)
        if front is None:
            row["reject_reason"] = "frontend_failed"

            poses_wc.append(poses_wc[-1].copy())
            timestamps.append(timestamp_from_path(f2))
            odometry_edges.append({
                "i": i,
                "j": i + 1,
                "z": np.array([0.0, 0.0, 0.0], dtype=float),
                "type": "odo"
            })
            debug_rows.append(row)
            rejected_count += 1

            print(
                f"[{i:04d}->{i+1:04d}] REJECT frontend_failed"
            )
            continue

        row["raw_matches"] = len(front["raw_matches"])
        row["filtered_matches"] = len(front["filtered_matches"])

        pose, pose_debug = process_frame_pair_pose(
            front["A"],
            front["B"],
            dataset_cfg.intrinsics,
            str(d1) if d1 is not None else None,
            str(d2) if d2 is not None else None,
            dataset_cfg.depth_scale,
            odom_cfg.min_essential_inliers,
            odom_cfg.min_metric_points,
        )

        row["essential_inliers"] = pose_debug["num_essential_inliers"]
        row["metric_points"] = pose_debug["num_metric_points"]

        if pose is None:
            row["reject_reason"] = pose_debug["reject_reason"]

            poses_wc.append(poses_wc[-1].copy())
            timestamps.append(timestamp_from_path(f2))
            odometry_edges.append({
                "i": i,
                "j": i + 1,
                "z": np.array([0.0, 0.0, 0.0], dtype=float),
                "type": "odo"
            })
            debug_rows.append(row)
            rejected_count += 1

            print(
                f"[{i:04d}->{i+1:04d}] REJECT {row['reject_reason']} | "
                f"raw={row['raw_matches']} filtered={row['filtered_matches']} "
                f"E_inliers={row['essential_inliers']} "
                f"metric_pts={row['metric_points']}"
            )
            continue

        if not pose["metric"]:
            row["reject_reason"] = pose_debug["reject_reason"] or "not_metric"

            poses_wc.append(poses_wc[-1].copy())
            timestamps.append(timestamp_from_path(f2))
            odometry_edges.append({
                "i": i,
                "j": i + 1,
                "z": np.array([0.0, 0.0, 0.0], dtype=float),
                "type": "odo"
            })
            debug_rows.append(row)
            rejected_count += 1

            print(
                f"[{i:04d}->{i+1:04d}] REJECT {row['reject_reason']} | "
                f"raw={row['raw_matches']} filtered={row['filtered_matches']} "
                f"E_inliers={row['essential_inliers']} "
                f"metric_pts={row['metric_points']}"
            )
            continue

        step_norm = np.linalg.norm(pose["t"].reshape(3))
        row["step_norm"] = float(step_norm)

        if step_norm > odom_cfg.max_step_metres:
            row["reject_reason"] = "step_too_large"

            poses_wc.append(poses_wc[-1].copy())
            timestamps.append(timestamp_from_path(f2))
            odometry_edges.append({
                "i": i,
                "j": i + 1,
                "z": np.array([0.0, 0.0, 0.0], dtype=float),
                "type": "odo"
            })
            debug_rows.append(row)
            rejected_count += 1

            print(
                f"[{i:04d}->{i+1:04d}] REJECT step_too_large | "
                f"raw={row['raw_matches']} filtered={row['filtered_matches']} "
                f"E_inliers={row['essential_inliers']} "
                f"metric_pts={row['metric_points']} "
                f"step={step_norm:.4f} m"
            )
            continue

        T_12 = make_T(pose["R"], pose["t"])
        T_wc_next = poses_wc[-1] @ np.linalg.inv(T_12)

        poses_wc.append(T_wc_next)
        timestamps.append(timestamp_from_path(f2))

        dx = float(pose["t"][0, 0])
        dz = float(pose["t"][2, 0])
        dth = yaw_from_R(pose["R"])

        odometry_edges.append({
            "i": i,
            "j": i + 1,
            "z": np.array([dx, dz, dth], dtype=float),
            "type": "odo"
        })

        row["accepted"] = True
        row["reject_reason"] = None
        debug_rows.append(row)
        accepted_count += 1

        print(
            f"[{i:04d}->{i+1:04d}] ACCEPT | "
            f"raw={row['raw_matches']} filtered={row['filtered_matches']} "
            f"E_inliers={row['essential_inliers']} "
            f"metric_pts={row['metric_points']} "
            f"step={step_norm:.4f} m"
        )

    poses_wc = np.stack(poses_wc, axis=0)
    timestamps = np.array(timestamps, dtype=float)

    np.savez(
        odom_cfg.output_dir / "poses_3d.npz",
        poses_wc=poses_wc,
        timestamps=timestamps,
        image_files=np.array([str(p) for p in rgb_files], dtype=object)
    )

    timestamp_str = f"{datetime.now():%Y%m%d_%H%M%S}"
    debug_path = odom_cfg.output_dir / f"odometry_debug_{timestamp_str}.csv"
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(
            "frame_i,frame_j,timestamp_i,timestamp_j,"
            "raw_matches,filtered_matches,essential_inliers,metric_points,"
            "step_norm,accepted,reject_reason\n"
        )
        for r in debug_rows:
            f.write(
                f"{r['frame_i']},{r['frame_j']},"
                f"{r['timestamp_i']:.6f},{r['timestamp_j']:.6f},"
                f"{r['raw_matches']},{r['filtered_matches']},"
                f"{r['essential_inliers']},{r['metric_points']},"
                f"{r['step_norm']:.6f},{int(r['accepted'])},"
                f"{r['reject_reason']}\n"
            )

    print("\n=== ODOMETRY SUMMARY ===")
    print(f"Total frame pairs: {len(rgb_files) - 1}")
    print(f"Accepted updates : {accepted_count}")
    print(f"Rejected updates : {rejected_count}")
    print(f"Saved debug CSV  : {debug_path}")

    return {
        "poses_wc": poses_wc,
        "timestamps": timestamps,
        "image_files": rgb_files,
        "edges": odometry_edges,
        "debug_rows": debug_rows,
    }
