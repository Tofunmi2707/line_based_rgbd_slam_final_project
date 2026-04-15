from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from src.pose_graph_2d import optimise_pose_graph_with_metrics


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) > 0 else float("nan")


def safe_median(x: np.ndarray) -> float:
    return float(np.median(x)) if len(x) > 0 else float("nan")


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
    plt.ylabel("y (m)")
    plt.title("Trajectory before and after loop closure")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def run_loop_closure_analysis(
    poses_xytheta: np.ndarray,
    edges: list[dict],
    output_dir: Path,
    gt_xy: np.ndarray | None = None,
    iters: int = 10,
    w_odo: float = 1.0,
    w_loop: float = 3.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    pg_out = optimise_pose_graph_with_metrics(
        poses_xytheta,
        edges,
        iters=iters,
        w_odo=w_odo,
        w_loop=w_loop,
    )

    odom_before = pg_out["odom_residuals_before"]
    odom_after = pg_out["odom_residuals_after"]
    loop_before = pg_out["loop_residuals_before"]
    loop_after = pg_out["loop_residuals_after"]

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

    save_loop_closure_trajectory_plot(
        output_dir / "loop_closure_trajectory_before_after.png",
        pg_out["poses_before"][:, :2],
        pg_out["poses_after"][:, :2],
        gt_xy,
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
        gt_xy=np.asarray(gt_xy, dtype=float)
        if gt_xy is not None else np.empty((0, 2)),
    )

    return {
        "poses_after": pg_out["poses_after"],
        "num_odom_edges": int(pg_out["num_odom_edges"]),
        "num_loop_edges": int(pg_out["num_loop_edges"]),
        "loop_mean_before": safe_mean(loop_before),
        "loop_mean_after": safe_mean(loop_after),
        "loop_median_before": safe_median(loop_before),
        "loop_median_after": safe_median(loop_after),
        "odom_mean_before": safe_mean(odom_before),
        "odom_mean_after": safe_mean(odom_after),
    }
