from __future__ import annotations

# Inspiration: NumPy-based summary statistics and matplotlib export;
# project-specific loop-closure residual reporting and dissertation summary
# outputs implemented for this project.

from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) > 0 else float("nan")


def safe_median(x: np.ndarray) -> float:
    return float(np.median(x)) if len(x) > 0 else float("nan")


def save_residual_table(
    out_csv: Path,
    odom_before: np.ndarray,
    odom_after: np.ndarray,
    loop_before: np.ndarray,
    loop_after: np.ndarray,
) -> None:
    # Inspiration: examiner-facing CSV table export; columns chosen to match the
    # loop-closure residual table used in the dissertation.
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
        writer.writerow(
            [
                "edge_type",
                "edge_count",
                "mean_residual_before",
                "median_residual_before",
                "mean_residual_after",
                "median_residual_after",
            ]
        )
        writer.writerows(rows)


def save_summary_txt(
    out_txt: Path,
    num_odom_edges: int,
    num_loop_edges: int,
    odom_before: np.ndarray,
    odom_after: np.ndarray,
    loop_before: np.ndarray,
    loop_after: np.ndarray,
) -> None:
    # Inspiration: plain-text audit trail for appendix/reproducibility.
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Loop closure summary\n")
        f.write("--------------------\n")
        f.write(f"Number of odometry edges: {num_odom_edges}\n")
        f.write(f"Number of loop edges: {num_loop_edges}\n\n")

        f.write("Odometry residuals\n")
        f.write(f"  Mean before  : {safe_mean(odom_before):.6f}\n")
        f.write(f"  Median before: {safe_median(odom_before):.6f}\n")
        f.write(f"  Mean after   : {safe_mean(odom_after):.6f}\n")
        f.write(f"  Median after : {safe_median(odom_after):.6f}\n\n")

        f.write("Loop residuals\n")
        f.write(f"  Mean before  : {safe_mean(loop_before):.6f}\n")
        f.write(f"  Median before: {safe_median(loop_before):.6f}\n")
        f.write(f"  Mean after   : {safe_mean(loop_after):.6f}\n")
        f.write(f"  Median after : {safe_median(loop_after):.6f}\n")


def save_before_after_residual_bar(
    out_png: Path,
    odom_before: np.ndarray,
    odom_after: np.ndarray,
    loop_before: np.ndarray,
    loop_after: np.ndarray,
) -> None:
    # Inspiration: summary bar chart; project-specific grouping chosen to show
    # the relative effect of loop closure on odometry and loop constraints.
    labels = [
        "Odometry mean\nbefore",
        "Odometry mean\nafter",
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


def save_before_after_trajectory_plot(
    out_png: Path,
    poses_before: np.ndarray,
    poses_after: np.ndarray,
    gt_xy: np.ndarray | None = None,
) -> None:
    # Inspiration: standard before/after trajectory visualisation; ground-truth
    # overlay included when available for supporting interpretation.
    plt.figure(figsize=(8, 6))

    if gt_xy is not None and len(gt_xy) > 0:
        plt.plot(gt_xy[:, 0], gt_xy[:, 1], label="Ground truth", linewidth=2)

    plt.plot(
        poses_before[:, 0],
        poses_before[:, 1],
        label="Before loop closure",
        linewidth=2,
    )
    plt.plot(
        poses_after[:, 0],
        poses_after[:, 1],
        label="After loop closure",
        linewidth=2,
    )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajectory before and after loop closure")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def load_metrics_file(npz_path: Path) -> dict:
    # Inspiration: saved .npz metrics workflow used elsewhere in the project.
    data = np.load(npz_path, allow_pickle=True)

    out = {
        "odom_before": np.asarray(data["odom_residuals_before"], dtype=float),
        "odom_after": np.asarray(data["odom_residuals_after"], dtype=float),
        "loop_before": np.asarray(data["loop_residuals_before"], dtype=float),
        "loop_after": np.asarray(data["loop_residuals_after"], dtype=float),
        "poses_before": np.asarray(data["poses_before_xy"], dtype=float),
        "poses_after": np.asarray(data["poses_after_xy"], dtype=float),
        "num_odom_edges": int(data["num_odom_edges"]),
        "num_loop_edges": int(data["num_loop_edges"]),
    }

    if "gt_xy" in data:
        out["gt_xy"] = np.asarray(data["gt_xy"], dtype=float)
    else:
        out["gt_xy"] = None

    return out


def find_latest_loop_metrics(results_dir: Path) -> tuple[Path, str, str] | None:
    # Inspiration: project-specific results layout under
    # results/<dataset>/<method>/loop_closure/...
    candidates: list[tuple[Path, str, str]] = []

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name in {"odometry_summary", "gifs", "calibration"}:
            continue

        for method_dir in sorted(dataset_dir.iterdir()):
            if not method_dir.is_dir():
                continue

            direct_npz = method_dir / "loop_closure_metrics.npz"
            if direct_npz.exists():
                candidates.append((direct_npz, dataset_dir.name, method_dir.name))

            nested_npz = method_dir / "loop_closure" / "loop_closure_metrics.npz"
            if nested_npz.exists():
                candidates.append((nested_npz, dataset_dir.name, method_dir.name))

    if not candidates:
        return None

    preferred = [
        ("fr2_large_with_loop", "v2_lbd_endpoints"),
        ("fr2_large_with_loop", "v3_geom_filter"),
        ("fr2_large_with_loop", "v1_centroid"),
    ]
    for pref_dataset, pref_method in preferred:
        for path, dataset, method in candidates:
            if dataset == pref_dataset and method == pref_method:
                return path, dataset, method

    return candidates[0]


def main() -> None:
    found = find_latest_loop_metrics(RESULTS_DIR)
    if found is None:
        print(
            "No loop_closure_metrics.npz file found under results/. "
            "Run your loop-closure experiment first."
        )
        sys.exit(1)

    npz_path, dataset_name, method_name = found
    metrics = load_metrics_file(npz_path)

    run_dir = npz_path.parent if npz_path.parent.name != method_name else npz_path.parent
    out_dir = run_dir / "loop_closure_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_residual_table(
        out_dir / "loop_closure_residual_table.csv",
        metrics["odom_before"],
        metrics["odom_after"],
        metrics["loop_before"],
        metrics["loop_after"],
    )

    save_summary_txt(
        out_dir / "loop_closure_summary.txt",
        metrics["num_odom_edges"],
        metrics["num_loop_edges"],
        metrics["odom_before"],
        metrics["odom_after"],
        metrics["loop_before"],
        metrics["loop_after"],
    )

    save_before_after_residual_bar(
        out_dir / "loop_closure_residuals_before_after.png",
        metrics["odom_before"],
        metrics["odom_after"],
        metrics["loop_before"],
        metrics["loop_after"],
    )

    save_before_after_trajectory_plot(
        out_dir / "loop_closure_trajectory_before_after.png",
        metrics["poses_before"],
        metrics["poses_after"],
        metrics["gt_xy"],
    )

    print("Saved loop-closure summary outputs to:")
    print(out_dir)
    print(f"Dataset: {dataset_name}")
    print(f"Method : {method_name}")
    print(f"Number of odometry edges: {metrics['num_odom_edges']}")
    print(f"Number of loop edges: {metrics['num_loop_edges']}")
    print(f"Loop mean residual before: {safe_mean(metrics['loop_before']):.6f}")
    print(f"Loop mean residual after : {safe_mean(metrics['loop_after']):.6f}")
    print(f"Loop median residual before: {safe_median(metrics['loop_before']):.6f}")
    print(f"Loop median residual after : {safe_median(metrics['loop_after']):.6f}")


if __name__ == "__main__":
    main()
