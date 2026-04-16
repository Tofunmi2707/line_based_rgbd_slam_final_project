from __future__ import annotations

# Inspiration: NumPy trajectory loading and matplotlib plotting; project-specific
# before/after and benchmark-comparison figure generation implemented here.

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_npz_array(npz_path: Path, key: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {npz_path}")
    return np.asarray(data[key])


def save_loop_closure_side_by_side(
    poses_before: np.ndarray,
    poses_after: np.ndarray,
    out_path: Path,
) -> None:
    # Inspiration: before/after trajectory comparison for loop-closure reporting.
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
    axes[0].set_ylabel("y position (m)")
    axes[0].grid(True)
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)

    axes[1].plot(poses_after[:, 0], poses_after[:, 1], linewidth=2.2)
    axes[1].scatter(poses_after[0, 0], poses_after[0, 1], s=70, marker="o", label="Start")
    axes[1].scatter(poses_after[-1, 0], poses_after[-1, 1], s=70, marker="X", label="End")
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
    # Inspiration: trajectory correction magnitude plot for backend interpretation.
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


def main() -> None:
    # Fastest use case: reuse already-generated loop-closure metrics if present.
    dataset_name = "fr2_large_with_loop"
    method_name = "v2_lbd_endpoints"

    npz_candidates = [
        PROJECT_ROOT / "results" / dataset_name / method_name / "loop_closure_metrics.npz",
        PROJECT_ROOT / "results" / dataset_name / method_name / "loop_closure" / "loop_closure_metrics.npz",
    ]

    npz_path = None
    for candidate in npz_candidates:
        if candidate.exists():
            npz_path = candidate
            break

    if npz_path is None:
        raise FileNotFoundError(
            "Could not find loop_closure_metrics.npz in the expected results folders."
        )

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

    print(f"Saved trajectory plots to: {out_dir}")


if __name__ == "__main__":
    main()
