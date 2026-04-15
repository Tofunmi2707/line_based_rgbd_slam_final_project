from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from config import DATASETS, PipelineConfig
from src.odometry import run_visual_odometry
from src.tum_io import (
    build_depth_index,
    nearest_depth_path,
    load_groundtruth,
    associate_gt_positions,
)
from src.line_frontend_v2_lbd_endpoints import process_frame_pair_frontend
from src.pose_estimation import process_frame_pair_pose
from src.pose_graph_2d import optimise_pose_graph_with_metrics


def yaw_from_R(R: np.ndarray) -> float:
    return float(np.arctan2(R[0, 2], R[2, 2]))


def poses_wc_to_xytheta(poses_wc: np.ndarray) -> np.ndarray:
    poses_2d = []
    for T in poses_wc:
        x = float(T[0, 3])
        z = float(T[2, 3])
        th = yaw_from_R(T[:3, :3])
        poses_2d.append([x, z, th])
    return np.asarray(poses_2d, dtype=float)


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) > 0 else float("nan")


def safe_median(x: np.ndarray) -> float:
    return float(np.median(x)) if len(x) > 0 else float("nan")


def find_loop_candidates(
    poses_xytheta: np.ndarray,
    min_frame_gap: int,
    pose_radius: float,
    max_candidates_per_frame: int = 1,
):
    candidates = []

    xy = poses_xytheta[:, :2]

    for j in range(len(poses_xytheta)):
        valid = []
        for i in range(0, j - min_frame_gap):
            d = float(np.linalg.norm(xy[j] - xy[i]))
            if d <= pose_radius:
                valid.append((d, i, j))

        valid.sort(key=lambda t: t[0])
        candidates.extend(valid[:max_candidates_per_frame])

    return candidates


def build_loop_edges(
    dataset_cfg,
    pipe_cfg,
    image_files,
    poses_wc,
    min_frame_gap: int = 80,
    pose_radius: float = 0.40,
    max_candidates_per_frame: int = 1,
    max_loop_step_metres: float = 1.0,
):
    poses_xytheta = poses_wc_to_xytheta(poses_wc)
    candidates = find_loop_candidates(
        poses_xytheta,
        min_frame_gap=min_frame_gap,
        pose_radius=pose_radius,
        max_candidates_per_frame=max_candidates_per_frame,
    )

    depth_ts, depth_files = build_depth_index(dataset_cfg.depth_dir)

    loop_edges = []
    loop_debug_rows = []

    for dist_2d, i, j in candidates:
        f1 = image_files[i]
        f2 = image_files[j]

        d1 = nearest_depth_path(
            f1, depth_ts, depth_files, dataset_cfg.max_rgb_depth_dt
        )
        d2 = nearest_depth_path(
            f2, depth_ts, depth_files, dataset_cfg.max_rgb_depth_dt
        )

        row = {
            "i": i,
            "j": j,
            "pose_distance_2d": dist_2d,
            "raw_matches": 0,
            "filtered_matches": 0,
            "essential_inliers": 0,
            "metric_points": 0,
            "step_norm": 0.0,
            "accepted": False,
            "reject_reason": None,
        }

        front = process_frame_pair_frontend(str(f1), str(f2), pipe_cfg)
        if front is None:
            row["reject_reason"] = "frontend_failed"
            loop_debug_rows.append(row)
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
            pipe_cfg.min_essential_inliers,
            pipe_cfg.min_metric_points,
        )

        row["essential_inliers"] = pose_debug["num_essential_inliers"]
        row["metric_points"] = pose_debug["num_metric_points"]

        if pose is None:
            row["reject_reason"] = pose_debug["reject_reason"]
            loop_debug_rows.append(row)
            continue

        if not pose["metric"]:
            row["reject_reason"] = pose_debug["reject_reason"] or "not_metric"
            loop_debug_rows.append(row)
            continue

        step_norm = float(np.linalg.norm(pose["t"].reshape(3)))
        row["step_norm"] = step_norm

        if step_norm > max_loop_step_metres:
            row["reject_reason"] = "step_too_large"
            loop_debug_rows.append(row)
            continue

        dx = float(pose["t"][0, 0])
        dz = float(pose["t"][2, 0])
        dth = yaw_from_R(pose["R"])

        loop_edges.append({
            "i": i,
            "j": j,
            "z": np.array([dx, dz, dth], dtype=float),
            "type": "loop",
        })

        row["accepted"] = True
        row["reject_reason"] = None
        loop_debug_rows.append(row)

    return loop_edges, loop_debug_rows


def save_loop_debug_csv(out_path: Path, rows: list[dict]):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "i", "j", "pose_distance_2d", "raw_matches", "filtered_matches",
            "essential_inliers", "metric_points", "step_norm",
            "accepted", "reject_reason"
        ])
        for r in rows:
            writer.writerow([
                r["i"],
                r["j"],
                f"{r['pose_distance_2d']:.6f}",
                r["raw_matches"],
                r["filtered_matches"],
                r["essential_inliers"],
                r["metric_points"],
                f"{r['step_norm']:.6f}",
                int(r["accepted"]),
                r["reject_reason"],
            ])


def save_loop_closure_residual_table(
    out_csv: Path,
    odom_before: np.ndarray,
    odom_after: np.ndarray,
    loop_before: np.ndarray,
    loop_after: np.ndarray,
):
    rows = [
        [
            "odometry",
            len(odom_before),
            safe_mean(odom_before),
            safe_median(odom_before),
            safe_mean(odom_after),
            safe_median(odom_after),
        ],
        [
            "loop",
            len(loop_before),
            safe_mean(loop_before),
            safe_median(loop_before),
            safe_mean(loop_after),
            safe_median(loop_after),
        ],
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "edge_type",
            "edge_count",
            "mean_residual_before",
            "median_residual_before",
            "mean_residual_after",
            "median_residual_after",
        ])
        writer.writerows(rows)


def save_loop_closure_summary_txt(
    out_txt: Path,
    num_odom_edges: int,
    num_loop_edges: int,
    odom_before: np.ndarray,
    odom_after: np.ndarray,
    loop_before: np.ndarray,
    loop_after: np.ndarray,
):
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Loop closure summary\n")
        f.write("--------------------\n")
        f.write(f"Number of odometry edges: {num_odom_edges}\n")
        f.write(f"Number of loop edges: {num_loop_edges}\n\n")

        f.write("Odometry residuals\n")
        f.write(f"  Mean before   : {safe_mean(odom_before):.6f}\n")
        f.write(f"  Median before : {safe_median(odom_before):.6f}\n")
        f.write(f"  Mean after    : {safe_mean(odom_after):.6f}\n")
        f.write(f"  Median after  : {safe_median(odom_after):.6f}\n\n")

        f.write("Loop residuals\n")
        f.write(f"  Mean before   : {safe_mean(loop_before):.6f}\n")
        f.write(f"  Median before : {safe_median(loop_before):.6f}\n")
        f.write(f"  Mean after    : {safe_mean(loop_after):.6f}\n")
        f.write(f"  Median after  : {safe_median(loop_after):.6f}\n")


def save_loop_closure_residual_plot(
    out_png: Path,
    odom_before: np.ndarray,
    odom_after: np.ndarray,
    loop_before: np.ndarray,
    loop_after: np.ndarray,
):
    labels = [
        "Odom mean\nbefore",
        "Odom mean\nafter",
        "Loop mean\nbefore",
        "Loop mean\nafter",
    ]
    values = [
        safe_mean(odom_before),
        safe_mean(odom_after),
        safe_mean(loop_before),
        safe_mean(loop_after),
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, edgecolor="black", linewidth=0.8)
    plt.ylabel("Mean residual")
    plt.title("Residuals before and after loop-closure optimisation")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_loop_closure_trajectory_plot(
    out_png: Path,
    poses_before_xy: np.ndarray,
    poses_after_xy: np.ndarray,
    gt_xy: np.ndarray | None = None,
):
    plt.figure(figsize=(7, 6))

    if gt_xy is not None and len(gt_xy) > 0:
        plt.plot(gt_xy[:, 0], gt_xy[:, 1], label="Ground truth", linewidth=2)

    plt.plot(
        poses_before_xy[:, 0],
        poses_before_xy[:, 1],
        label="Before loop closure",
        linewidth=2,
    )
    plt.plot(
        poses_after_xy[:, 0],
        poses_after_xy[:, 1],
        label="After loop closure",
        linewidth=2,
    )

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Trajectory before and after loop closure")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def build_gt_xz(dataset_cfg, timestamps: np.ndarray) -> np.ndarray:
    gt_t, gt_xyz = load_groundtruth(dataset_cfg.groundtruth_path)
    gt_interp = associate_gt_positions(timestamps, gt_t, gt_xyz)
    gt_interp = gt_interp - gt_interp[0]
    return gt_interp[:, [0, 2]]


def run_loop_closure_stage(
    dataset_cfg,
    odom_cfg,
    loop_cfg,
    poses_wc: np.ndarray,
    timestamps: np.ndarray,
    image_files,
    odometry_edges: list[dict],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    loop_edges, loop_debug_rows = build_loop_edges(
        dataset_cfg=dataset_cfg,
        pipe_cfg=odom_cfg,
        image_files=image_files,
        poses_wc=poses_wc,
        min_frame_gap=loop_cfg.min_frame_gap,
        pose_radius=loop_cfg.pose_radius,
        max_candidates_per_frame=loop_cfg.max_candidates_per_frame,
        max_loop_step_metres=loop_cfg.max_loop_step_metres,
    )

    all_edges = list(odometry_edges) + loop_edges
    poses_xytheta = poses_wc_to_xytheta(poses_wc)

    pg_out = optimise_pose_graph_with_metrics(
        poses_xytheta,
        all_edges,
        iters=loop_cfg.iters,
        w_odo=loop_cfg.w_odo,
        w_loop=loop_cfg.w_loop,
    )

    odom_before = pg_out["odom_residuals_before"]
    odom_after = pg_out["odom_residuals_after"]
    loop_before = pg_out["loop_residuals_before"]
    loop_after = pg_out["loop_residuals_after"]

    save_loop_debug_csv(output_dir / "loop_edge_debug.csv", loop_debug_rows)

    save_loop_closure_residual_table(
        output_dir / "loop_closure_residual_table.csv",
        odom_before,
        odom_after,
        loop_before,
        loop_after,
    )

    save_loop_closure_summary_txt(
        output_dir / "loop_closure_summary.txt",
        pg_out["num_odom_edges"],
        pg_out["num_loop_edges"],
        odom_before,
        odom_after,
        loop_before,
        loop_after,
    )

    save_loop_closure_residual_plot(
        output_dir / "loop_closure_residuals_before_after.png",
        odom_before,
        odom_after,
        loop_before,
        loop_after,
    )

    gt_xz = build_gt_xz(dataset_cfg, timestamps)

    save_loop_closure_trajectory_plot(
        output_dir / "loop_closure_trajectory_before_after.png",
        pg_out["poses_before"][:, :2],
        pg_out["poses_after"][:, :2],
        gt_xz,
    )

    np.savez(
        output_dir / "loop_closure_metrics.npz",
        odom_residuals_before=odom_before,
        odom_residuals_after=odom_after,
        loop_residuals_before=loop_before,
        loop_residuals_after=loop_after,
        poses_before_xy=pg_out["poses_before"][:, :2],
        poses_after_xy=pg_out["poses_after"][:, :2],
        num_odom_edges=int(pg_out["num_odom_edges"]),
        num_loop_edges=int(pg_out["num_loop_edges"]),
        gt_xy=gt_xz,
    )

    print("\n=== LOOP CLOSURE SUMMARY ===")
    print(f"Number of odometry edges: {pg_out['num_odom_edges']}")
    print(f"Number of loop edges    : {pg_out['num_loop_edges']}")
    print(f"Loop mean residual before : {safe_mean(loop_before):.6f}")
    print(f"Loop mean residual after  : {safe_mean(loop_after):.6f}")
    print(f"Loop median residual before: {safe_median(loop_before):.6f}")
    print(f"Loop median residual after : {safe_median(loop_after):.6f}")
    print(f"Saved loop-closure outputs to: {output_dir}")

    return {
        "poses_before_xytheta": pg_out["poses_before"],
        "poses_after_xytheta": pg_out["poses_after"],
        "loop_edges": loop_edges,
        "num_odom_edges": int(pg_out["num_odom_edges"]),
        "num_loop_edges": int(pg_out["num_loop_edges"]),
        "loop_mean_before": safe_mean(loop_before),
        "loop_mean_after": safe_mean(loop_after),
        "loop_median_before": safe_median(loop_before),
        "loop_median_after": safe_median(loop_after),
    }


def main():
    dataset_name = "fr2_large_with_loop"
    method_name = "v2_lbd_endpoints"

    dataset_cfg = DATASETS[dataset_name]

    pipe_cfg = PipelineConfig()
    pipe_cfg.output_dir = Path("results") / dataset_name / method_name

    print(f"Running visual odometry for {dataset_name}...")
    odo_out = run_visual_odometry(dataset_cfg, pipe_cfg)

    print("\nRunning loop closure...")
    lc_out = run_loop_closure_stage(
        dataset_cfg=dataset_cfg,
        pipe_cfg=pipe_cfg,
        poses_wc=odo_out["poses_wc"],
        timestamps=odo_out["timestamps"],
        image_files=odo_out["image_files"],
        odometry_edges=odo_out["edges"],
        output_dir=pipe_cfg.output_dir,
        min_frame_gap=80,
        pose_radius=0.40,
        max_candidates_per_frame=1,
        max_loop_step_metres=1.0,
        iters=10,
        w_odo=1.0,
        w_loop=3.0,
    )

    print("\nLoop closure completed.")
    print(f"Accepted loop edges: {lc_out['num_loop_edges']}")
    print(f"Saved outputs to: {pipe_cfg.output_dir}")


if __name__ == "__main__":
    main()
