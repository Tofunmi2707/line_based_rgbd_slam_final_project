from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.tum_io import load_groundtruth, associate_gt_positions


def umeyama_similarity_alignment(src: np.ndarray, dst: np.ndarray):
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
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def save_trajectory_plots(gt_xyz, est_xyz_aligned, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="Ground truth")
    ax.plot(est_xyz_aligned[:, 0],
            est_xyz_aligned[:, 1],
            est_xyz_aligned[:, 2],
            label="Estimated aligned")
    ax.set_title("Trajectory comparison (3D)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_3d.png", dpi=300)
    plt.close()

    pairs = [("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)]
    for name, i, j in pairs:
        plt.figure(figsize=(6, 6))
        plt.plot(gt_xyz[:, i], gt_xyz[:, j], label="Ground truth")
        plt.plot(est_xyz_aligned[:, i],
                 est_xyz_aligned[:, j],
                 label="Estimated aligned")
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel(["x", "y", "z"][i] + " (m)")
        plt.ylabel(["x", "y", "z"][j] + " (m)")
        plt.title(f"Trajectory comparison ({name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"trajectory_{name}.png", dpi=300)
        plt.close()


def evaluate_trajectory(dataset_cfg,
                        poses_wc: np.ndarray,
                        timestamps: np.ndarray,
                        output_dir: Path):
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
        rmse=rmse
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
