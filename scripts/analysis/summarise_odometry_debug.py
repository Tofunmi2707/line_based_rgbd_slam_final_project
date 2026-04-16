from __future__ import annotations

"""
Summarise odometry debug outputs across datasets and pipeline variants.

This script reads the saved odometry debug CSV files and trajectory evaluation
outputs produced by the main pipeline, then generates summary tables and
diagnostic plots for use in the dissertation.

It is used to:
- aggregate accepted and rejected frame-pair counts,
- compute acceptance rates and average diagnostic quantities,
- load per-run RMSE values,
- create per-dataset comparison tables,
- export a rejection-reason bar chart for the baseline failure analysis.

Inspiration:
- Standard CSV-based result aggregation and matplotlib chart generation.
- The aggregation metrics, dataset/method grouping, and rejection-diagnosis
  outputs were designed within the present project to support the odometry
  comparison and failure-analysis sections of the report.
"""

from pathlib import Path
import sys
import csv
from collections import Counter

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_DIR = RESULTS_DIR / "odometry_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_debug_csv(run_dir: Path) -> Path | None:
    """
    Find the most recent odometry debug CSV in a run directory.

    Args:
        run_dir: Directory containing odometry outputs for one dataset/method run.

    Returns:
        Path to the latest matching debug CSV, or None if no debug CSV exists.
    """
    csv_files = sorted(run_dir.glob("odometry_debug_*.csv"))
    if not csv_files:
        return None
    return csv_files[-1]


def safe_float(x, default: float = 0.0) -> float:
    """
    Convert a value to float safely.

    Args:
        x: Input value.
        default: Value returned if conversion fails.

    Returns:
        Converted float or the default value.
    """
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default: int = 0) -> int:
    """
    Convert a value to int safely.

    Args:
        x: Input value.
        default: Value returned if conversion fails.

    Returns:
        Converted integer or the default value.
    """
    try:
        return int(x)
    except Exception:
        return default


def read_debug_csv(csv_path: Path) -> dict:
    """
    Read one odometry debug CSV and compute summary statistics.

    The returned metrics are chosen to support the dissertation tables and
    odometry failure analysis.

    Args:
        csv_path: Path to the odometry debug CSV.

    Returns:
        Dictionary containing counts, averages, rejection-reason frequencies,
        and the raw loaded rows.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    accepted = sum(safe_int(r.get("accepted", 0)) for r in rows)
    rejected = total - accepted

    reject_counter = Counter()
    for r in rows:
        if safe_int(r.get("accepted", 0)) == 0:
            reason = (r.get("reject_reason") or "unknown").strip()
            reject_counter[reason] += 1

    raw_matches = [safe_int(r.get("raw_matches", 0)) for r in rows]
    filtered_matches = [safe_int(r.get("filtered_matches", 0)) for r in rows]
    essential_inliers = [safe_int(r.get("essential_inliers", 0)) for r in rows]
    metric_points = [safe_int(r.get("metric_points", 0)) for r in rows]
    step_norms = [safe_float(r.get("step_norm", 0.0)) for r in rows]

    acceptance_rate = 100.0 * accepted / total if total > 0 else 0.0

    return {
        "csv_path": csv_path,
        "rows": rows,
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": acceptance_rate,
        "reject_counter": reject_counter,
        "avg_raw_matches": (
            sum(raw_matches) / len(raw_matches) if raw_matches else 0.0
        ),
        "avg_filtered_matches": (
            sum(filtered_matches) / len(filtered_matches)
            if filtered_matches else 0.0
        ),
        "avg_essential_inliers": (
            sum(essential_inliers) / len(essential_inliers)
            if essential_inliers else 0.0
        ),
        "avg_metric_points": (
            sum(metric_points) / len(metric_points)
            if metric_points else 0.0
        ),
        "avg_step_norm": (
            sum(step_norms) / len(step_norms) if step_norms else 0.0
        ),
    }


def find_rmse(run_dir: Path) -> float | None:
    """
    Load the saved trajectory RMSE for a run if available.

    Args:
        run_dir: Directory containing one dataset/method run.

    Returns:
        RMSE value in metres, or None if it cannot be loaded.
    """
    npz_path = run_dir / "trajectory_eval.npz"
    if not npz_path.exists():
        return None

    try:
        import numpy as np
        data = np.load(npz_path, allow_pickle=True)
        return float(data["rmse"])
    except Exception:
        return None


def collect_runs(results_dir: Path) -> list[dict]:
    """
    Collect odometry summaries for all dataset/method runs under the results directory.

    The script assumes the directory convention:
    results/<dataset>/<method>/...

    Args:
        results_dir: Root results directory.

    Returns:
        List of run-summary dictionaries.
    """
    runs = []

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

            csv_path = find_latest_debug_csv(method_dir)
            if csv_path is None:
                continue

            summary = read_debug_csv(csv_path)
            summary["dataset"] = dataset_dir.name
            summary["method"] = method_dir.name
            summary["run_dir"] = method_dir
            summary["rmse"] = find_rmse(method_dir)
            runs.append(summary)

    return runs


def save_rejection_bar_chart(run: dict, out_path: Path) -> None:
    """
    Save a bar chart of rejection reasons for one run.

    Args:
        run: Run-summary dictionary.
        out_path: Output image path.

    Returns:
        None
    """
    counter = run["reject_counter"]
    if not counter:
        print(
            f"No rejection reasons found for {run['dataset']} / "
            f"{run['method']}"
        )
        return

    labels = list(counter.keys())
    values = [counter[k] for k in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, edgecolor="black", linewidth=0.8)
    plt.title(f"Rejection reasons: {run['dataset']} / {run['method']}")
    plt.xlabel("Reject reason")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_csv_table(path: Path, headers: list[str], rows: list[list]) -> None:
    """
    Write a CSV table to disk.

    Args:
        path: Output CSV path.
        headers: Header row.
        rows: Data rows.

    Returns:
        None
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def make_overall_counts_table(runs: list[dict], out_dir: Path) -> None:
    """
    Save a CSV table summarising all collected runs.

    Args:
        runs: List of run-summary dictionaries.
        out_dir: Output directory.

    Returns:
        None
    """
    headers = [
        "dataset",
        "method",
        "total_pairs",
        "accepted",
        "rejected",
        "acceptance_rate_percent",
        "avg_raw_matches",
        "avg_filtered_matches",
        "avg_essential_inliers",
        "avg_metric_points",
        "avg_step_norm",
        "rmse_m",
    ]
    rows = []

    for r in runs:
        rows.append(
            [
                r["dataset"],
                r["method"],
                r["total"],
                r["accepted"],
                r["rejected"],
                f"{r['acceptance_rate']:.2f}",
                f"{r['avg_raw_matches']:.2f}",
                f"{r['avg_filtered_matches']:.2f}",
                f"{r['avg_essential_inliers']:.2f}",
                f"{r['avg_metric_points']:.2f}",
                f"{r['avg_step_norm']:.4f}",
                "" if r["rmse"] is None else f"{r['rmse']:.4f}",
            ]
        )

    write_csv_table(
        out_dir / "accepted_rejected_counts_all_runs.csv",
        headers,
        rows,
    )


def make_dataset_comparison_table(
    runs: list[dict],
    dataset_name: str,
    out_dir: Path,
) -> None:
    """
    Save a CSV table comparing methods for one dataset.

    Args:
        runs: List of run-summary dictionaries.
        dataset_name: Dataset to summarise.
        out_dir: Output directory.

    Returns:
        None
    """
    dataset_runs = [r for r in runs if r["dataset"] == dataset_name]
    if not dataset_runs:
        return

    method_order = {
        "v1_centroid": 1,
        "v2_lbd_endpoints": 2,
        "v3_geom_filter": 3,
    }
    dataset_runs.sort(
        key=lambda r: (
            method_order.get(r["method"], 99),
            r["method"],
        )
    )

    headers = [
        "dataset",
        "method",
        "accepted",
        "rejected",
        "acceptance_rate_percent",
        "avg_raw_matches",
        "avg_filtered_matches",
        "avg_essential_inliers",
        "avg_metric_points",
        "rmse_m",
    ]
    rows = []

    for r in dataset_runs:
        rows.append(
            [
                r["dataset"],
                r["method"],
                r["accepted"],
                r["rejected"],
                f"{r['acceptance_rate']:.2f}",
                f"{r['avg_raw_matches']:.2f}",
                f"{r['avg_filtered_matches']:.2f}",
                f"{r['avg_essential_inliers']:.2f}",
                f"{r['avg_metric_points']:.2f}",
                "" if r["rmse"] is None else f"{r['rmse']:.4f}",
            ]
        )

    write_csv_table(out_dir / f"{dataset_name}_comparison.csv", headers, rows)


def choose_initial_run_for_rejection_chart(runs: list[dict]) -> dict | None:
    """
    Choose the most informative run for the baseline rejection-reason chart.

    The preferred order reflects the project reporting priority, with
    fr1_room / v1_centroid used first when available.

    Args:
        runs: List of run-summary dictionaries.

    Returns:
        Selected run-summary dictionary, or None if no runs exist.
    """
    preferred = [
        ("fr1_room", "v1_centroid"),
        ("fr1_desk", "v1_centroid"),
        ("fr1_xyz", "v1_centroid"),
    ]
    for dataset, method in preferred:
        for r in runs:
            if r["dataset"] == dataset and r["method"] == method:
                return r
    return runs[0] if runs else None


def print_markdown_tables(runs: list[dict]) -> None:
    """
    Print quick markdown tables for drafting and notes.

    Args:
        runs: List of run-summary dictionaries.

    Returns:
        None
    """
    print("\nOVERALL COUNTS TABLE\n")
    print(
        "| Dataset | Method | Accepted | Rejected | "
        "Acceptance rate (%) | RMSE (m) |"
    )
    print("|---|---|---:|---:|---:|---:|")
    for r in runs:
        rmse_str = "-" if r["rmse"] is None else f"{r['rmse']:.4f}"
        print(
            f"| {r['dataset']} | {r['method']} | {r['accepted']} | "
            f"{r['rejected']} | {r['acceptance_rate']:.2f} | {rmse_str} |"
        )

    for dataset_name in sorted({r["dataset"] for r in runs}):
        ds_runs = [r for r in runs if r["dataset"] == dataset_name]
        print(f"\n{dataset_name.upper()} COMPARISON TABLE\n")
        print(
            "| Method | Accepted | Rejected | Acceptance rate (%) | "
            "RMSE (m) |"
        )
        print("|---|---:|---:|---:|---:|")
        for r in ds_runs:
            rmse_str = "-" if r["rmse"] is None else f"{r['rmse']:.4f}"
            print(
                f"| {r['method']} | {r['accepted']} | {r['rejected']} | "
                f"{r['acceptance_rate']:.2f} | {rmse_str} |"
            )


def main() -> None:
    """
    Generate odometry summary tables and the main rejection-reason chart.

    Returns:
        None
    """
    runs = collect_runs(RESULTS_DIR)

    if not runs:
        print(f"No odometry debug CSVs found under: {RESULTS_DIR}")
        sys.exit(1)

    make_overall_counts_table(runs, SUMMARY_DIR)

    for dataset_name in sorted({r["dataset"] for r in runs}):
        make_dataset_comparison_table(runs, dataset_name, SUMMARY_DIR)

    chosen = choose_initial_run_for_rejection_chart(runs)
    if chosen is not None:
        save_rejection_bar_chart(
            chosen,
            SUMMARY_DIR / (
                f"rejection_reasons_{chosen['dataset']}_{chosen['method']}.png"
            ),
        )

    print_markdown_tables(runs)

    print("\nSaved summary outputs to:")
    print(SUMMARY_DIR)


if __name__ == "__main__":
    main()
