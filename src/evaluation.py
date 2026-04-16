from __future__ import annotations

"""
Trajectory evaluation utilities for the line-based RGB-D SLAM pipeline.

This module aligns the estimated trajectory to benchmark ground truth, computes
root mean square error after alignment, and saves the trajectory plots used in
the dissertation.

Inspiration:
- Similarity alignment follows the Umeyama least-squares method for aligning
  two point sets.
- Ground-truth association uses the project dataset I/O utilities for the TUM
  RGB-D benchmark.
- The plot generation and saved evaluation outputs were integrated within the
  present project for benchmark comparison and report production.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.tum_io import load_groundtruth, associate_gt_positions


def umeyama_similarity_alignment(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Align one 3D point set to another using a similarity transform.

    The alignment estimates:
    - a global scale factor,
    - a rotation matrix,
    - a translation vector,

    such that the transformed source points best match the destination points
    in the least-squares sense.

    Inspiration:
    - Umeyama similarity alignment for point-set registration.

    Args:
        src: Source point set with shape (N, 3).
        dst: Destination point set with shape (N, 3).

    Returns:
        Tuple containing:
        - aligned: Source points after similarity alignment,
        - scale: Estimated global scale factor,
        - R: Estimated 3 x 3 rotation matrix,
        - t: Estimated 3D translation vector.
    """
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    X = src - mu_src
    Y = dst - mu_dst

    C = (X.T @ Y) / len(src)
    U, S, Vt = np.linalg.svd(C)

    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    var_src = np.mean(np.sum(X**2, axis=1))
    scale = np.sum(S) / var_src
    t = mu_dst - scale * (R @ mu_src)

    aligned = (scale * (src @ R.T)) + t
    return aligned, scale, R, t


def compute_rmse(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute root mean square error between two aligned 3D trajectories.

    Args:
        A: First point set with shape (N, 3).
        B: Second point set with shape (N, 3).

    Returns:
        RMSE value in metres.
    """
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def save_trajectory_plots(
    gt_xyz: np.ndarray,
    est_xyz_aligned: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Save 3D and orthogonal trajectory comparison plots.

    The saved outputs include:
    - one 3D trajectory overlay,
    - three planar projections: xy, xz, and yz.

    Args:
        gt_xyz: Ground-truth trajectory points with shape (N, 3).
        est_xyz_aligned: Aligned estimated trajectory points with shape (N, 3).
        out_dir: Directory in which the plots will be saved.

    Returns:
        None
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="Ground truth")
    ax.plot(
        est_xyz_aligned[:, 0],
        est_xyz_aligned[:, 1],
        est_xyz_aligned[:, 2],
        label="Estimated aligned",
    )
    ax.set_title("Trajectory comparison (3D)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_3d.png", dpi=300)
    plt.close()

    pairs = [("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)]
    axis_names = ["x", "y", "z"]

    for name, i, j in pairs:
        plt.figure(figsize=(6, 6))
        plt.plot(gt_xyz[:, i], gt_xyz[:, j], label="Ground truth")
        plt.plot(
            est_xyz_aligned[:, i],
            est_xyz_aligned[:, j],
            label="Estimated aligned",
        )
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel(axis_names[i] + " (m)")
        plt.ylabel(axis_names[j] + " (m)")
        plt.title(f"Trajectory comparison ({name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"trajectory_{name}.png", dpi=300)
        plt.close()


def evaluate_trajectory(
    dataset_cfg,
    poses_wc: np.ndarray,
    timestamps: np.ndarray,
    output_dir: Path,
) -> dict:
    """
    Evaluate the estimated trajectory against benchmark ground truth.

    The function:
    1. extracts camera positions from the estimated poses,
    2. loads the benchmark ground truth,
    3. interpolates ground-truth positions to the estimated timestamps,
    4. normalises both trajectories to a common starting point,
    5. aligns the estimated trajectory using similarity alignment,
    6. computes RMSE,
    7. saves plots and evaluation data.

    Args:
        dataset_cfg: Dataset configuration containing the ground-truth path.
        poses_wc: World-frame camera poses with shape (N, 4, 4).
        timestamps: Timestamps aligned with the estimated poses.
        output_dir: Directory in which evaluation outputs should be saved.

    Returns:
        Dictionary containing:
        - rmse: Aligned root mean square error,
        - scale: Similarity-alignment scale factor,
        - gt_xyz: Interpolated ground-truth trajectory,
        - est_xyz: Unaligned estimated trajectory positions,
        - est_xyz_aligned: Similarity-aligned estimated trajectory.
    """
    est_xyz = poses_wc[:, :3, 3]

    gt_t, gt_xyz = load_groundtruth(dataset_cfg.groundtruth_path)
    gt_interp = associate_gt_positions(timestamps, gt_t, gt_xyz)

    est_xyz = est_xyz - est_xyz[0]
    gt_interp = gt_interp - gt_interp[0]

    est_aligned, scale, R, t = umeyama_similarity_alignment(est_xyz, gt_interp)
    rmse = compute_rmse(est_aligned, gt_interp)

    save_trajectory_plots(gt_interp, est_aligned, output_dir)

    np.savez(
        output_dir / "trajectory_eval.npz",
        timestamps=timestamps,
        gt_xyz=gt_interp,
        est_xyz=est_xyz,
        est_xyz_aligned=est_aligned,
        scale=scale,
        R=R,
        t=t,
        rmse=rmse,
    )

    print(f"Trajectory scale factor: {scale:.6f}")
    print(f"Trajectory RMSE: {rmse:.4f} m")

    return {
        "rmse": rmse,
        "scale": scale,
        "gt_xyz": gt_interp,
        "est_xyz": est_xyz,
        "est_xyz_aligned": est_aligned,
    }
