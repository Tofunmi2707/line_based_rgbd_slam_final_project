from __future__ import annotations

"""
Generate loop-closure trajectory plots for all available dataset/method runs.

This script searches the results directory for saved loop-closure metrics files
and generates:
- a side-by-side trajectory plot before vs after loop closure,
- a per-frame correction-magnitude plot,

for every available run.

Inspiration:
- NumPy trajectory loading and matplotlib plotting.
- The multi-run search and report-output structure were integrated within the
  present project so loop-closure figures can be regenerated across all
  available dataset/method combinations.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"


def save_loop_closure_side_by_side(
    poses_before: np.ndarray,
    poses_after: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save a side-by-side trajectory comparison before and after loop closure.

    Args:
        poses_before: Planar poses before loop closure, shape (N, 2) or (N, 3).
        poses_after: Planar poses after loop closure, shape (N, 2) or (N, 3).
        out_path: Output figure path.

    Returns:
        None
    """
    all_xy = np.vstack([poses_before[:, :2], poses_after[:, :2]])
    xmin, xmax = all_xy[:, 0].min(), all_xy[:, 0].max()
    ymin, ymax = all_xy[:, 1].min(), all_xy[:, 1].max()

    pad_x = 0.05 * max(1e-6, xmax - xmin)
    pad_y = 0.05 * max(1e-6, ymax - ymin)

    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    axes[0].plot(poses_before[:, 0], poses_before[:, 1], linewidth=2.2)
    axes[0].scatter(poses_before[0, 0], poses_before[0, 1], s=70, marker="o")
    axes[0].scatter(poses_before[-1, 0], poses_before[-1, 1], s=70, marker="X")
    axes[0].set_title("Trajectory before loop closure")
    axes[0].set_xlabel("x position (m)")
    axes[0].set_ylabel("z position (m)")
    axes[0].grid(True)
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)

    axes[1].plot(poses_after[:, 0], poses_after[:, 1], linewidth=2.2)
    axes[1].scatter(
        poses_after[0, 0],
        poses_after[0, 1],
        s=70,
        marker="o",
        label="Start",
    )
    axes[1].scatter(
        poses_after[-1, 0],
        poses_after[-1, 1],
        s=70,
        marker="X",
        label="End",
    )
    axes[1].set_title("Trajectory after loop closure")
    axes[1].set_xlabel("x position (m)")
    axes[1].grid(True)
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymin, ymax)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_correction_plot(
    poses_before: np.ndarray,
    poses_after: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save a plot of trajectory correction magnitude versus frame index.

    Args:
        poses_before: Planar poses before loop closure.
        poses_after: Planar poses after loop closure.
        out_path: Output figure path.

    Returns:
        None
    """
    corr = np.linalg.norm(poses_after[:, :2] - poses_before[:, :2], axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(corr, linewidth=2)
    plt.xlabel("Frame index")
    plt.ylabel("Pose correction magnitude (m)")
    plt.title("How much loop closure changed the trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def find_all_loop_metrics(results_dir: Path) -> list[tuple[Path, str, str]]:
    """
    Find all saved loop-closure metrics files under the results directory.

    The expected layouts are:
    - results/<dataset>/<method>/loop_closure_metrics.npz
    - results/<dataset>/<method>/loop_closure/loop_closure_metrics.npz

    Args:
        results_dir: Root results directory.

    Returns:
        List of tuples:
        - path to metrics file,
        - dataset name,
        - method name.
    """
    found: list[tuple[Path, str, str]] = []

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
                found.append((direct_npz, dataset_dir.name, method_dir.name))

            nested_npz = method_dir / "loop_closure" / "loop_closure_metrics.npz"  # noqa: E501
            if nested_npz.exists():
                found.append((nested_npz, dataset_dir.name, method_dir.name))

    return found


def main() -> None:
    """
    Generate loop-closure trajectory plots for every available run.

    Returns:
        None
    """
    runs = find_all_loop_metrics(RESULTS_DIR)

    if not runs:
        raise FileNotFoundError(
            "Could not find any loop_closure_metrics.npz files under results/."
        )

    for npz_path, dataset_name, method_name in runs:
        data = np.load(npz_path, allow_pickle=True)
        poses_before = np.asarray(data["poses_before_xy"], dtype=float)
        poses_after = np.asarray(data["poses_after_xy"], dtype=float)

        out_dir = npz_path.parent / "trajectory_plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        save_loop_closure_side_by_side(
            poses_before,
            poses_after,
            out_dir / "trajectory_side_by_side_clean.png",
        )

        save_correction_plot(
            poses_before,
            poses_after,
            out_dir / "trajectory_correction_vs_frame.png",
        )

        print(f"Saved trajectory plots for {dataset_name} / {method_name} to: {out_dir}")  # noqa: E501


if __name__ == "__main__":
    main()
