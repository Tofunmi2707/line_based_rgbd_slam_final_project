from pathlib import Path
import numpy as np


def timestamp_from_path(path: str | Path) -> float:
    path = Path(path)
    return float(path.stem)


def list_image_files(folder: Path, suffix: str = "*.png") -> list[Path]:
    files = sorted(folder.glob(suffix))
    if not files:
        raise RuntimeError(f"No files found in {folder}")
    return files


def build_depth_index(depth_dir: Path) -> tuple[np.ndarray, list[Path]]:
    depth_files = list_image_files(depth_dir, "*.png")
    depth_ts = np.array([timestamp_from_path(p) for p in depth_files],
                        dtype=float)
    return depth_ts, depth_files


def nearest_depth_path(
    rgb_path: Path,
    depth_ts: np.ndarray,
    depth_files: list[Path],
    max_dt: float,
) -> Path | None:
    t = timestamp_from_path(rgb_path)
    k = int(np.argmin(np.abs(depth_ts - t)))
    if abs(depth_ts[k] - t) > max_dt:
        return None
    return depth_files[k]


def load_groundtruth(gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    gt_all = np.loadtxt(gt_path, comments="#")
    t = gt_all[:, 0]
    xyz = gt_all[:, 1:4]
    idx = np.argsort(t)
    return t[idx], xyz[idx]


def associate_gt_positions(
    est_timestamps: np.ndarray,
    gt_timestamps: np.ndarray,
    gt_xyz: np.ndarray
) -> np.ndarray:
    x = np.interp(est_timestamps, gt_timestamps, gt_xyz[:, 0])
    y = np.interp(est_timestamps, gt_timestamps, gt_xyz[:, 1])
    z = np.interp(est_timestamps, gt_timestamps, gt_xyz[:, 2])
    return np.column_stack([x, y, z])


def load_rgb_sequence(rgb_dir: Path) -> list[Path]:
    return list_image_files(rgb_dir, "*.png")
