from __future__ import annotations

# Inspiration: standard CSV summary processing and matplotlib bar-chart export;
# project-specific aggregation, rejection-reason diagnosis, and dissertation
# table generation implemented for this project.

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
    # Inspiration: simple run-manifest style discovery; latest debug CSV is used
    # so the summary reflects the final run in each dataset/method folder.
    csv_files = sorted(run_dir.glob("odometry_debug_*.csv"))
    if not csv_files:
        return None
    return csv_files[-1]


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def read_debug_csv(csv_path: Path) -> dict:
    # Inspiration: CSV DictReader pattern; project-specific summary metrics chosen
    # to match the odometry tables and rejection-reason figures in the report.
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
    # Inspiration: saved NumPy evaluation files from the project pipeline;
    # RMSE is loaded here so summary tables align with the reported trajectory metric.
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
    # Inspiration: recursive folder summary workflow; project-specific directory
    # convention is results/<dataset>/<method>/...
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
    # Inspiration: matplotlib category bar charts; the rejection categories here
    # are the project’s diagnostic mechanism for explaining pipeline failure modes.
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
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def make_overall_counts_table(runs: list[dict], out_dir: Path) -> None:
    # Inspiration: examiner-facing summary table; columns chosen to match the
    # main odometry comparison section in the dissertation.
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
    # Inspiration: per-dataset variant table for direct V1/V2/V3 comparison.
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
    # Inspiration: project-specific reporting priority; fr1_room / v1_centroid
    # is preferred because it gives the clearest baseline failure diagnosis.
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
    # Inspiration: quick copy-paste markdown output for notes and report drafting.
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