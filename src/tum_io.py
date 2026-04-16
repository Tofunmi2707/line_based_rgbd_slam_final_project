from __future__ import annotations

"""
Utilities for loading TUM RGB-D image sequences and ground-truth
trajectory data.

This module provides helper functions for:
- extracting timestamps from TUM-style filenames,
- listing RGB and depth image files,
- finding the nearest depth frame for a given RGB frame,
- loading ground-truth trajectory positions,
- interpolating benchmark ground truth onto estimated timestamps.

Inspiration:
- The file layout and timestamp conventions follow the TUM RGB-D benchmark.
- RGB/depth pairing is based on nearest-timestamp association with a maximum
  allowed time difference, consistent with the dataset handling needs of the
  project.
- Ground-truth interpolation is included so estimated trajectories can be
  compared directly against benchmark reference positions.
"""

from pathlib import Path
import numpy as np


def timestamp_from_path(path: str | Path) -> float:
    """
    Extract a floating-point timestamp from a TUM-style image filename.

    Args:
        path: Path to an RGB or depth image file whose stem is a timestamp.

    Returns:
        Timestamp parsed from the filename stem.
    """
    path = Path(path)
    return float(path.stem)


def list_image_files(folder: Path, suffix: str = "*.png") -> list[Path]:
    """
    List image files in a folder using a given suffix pattern.

    Args:
        folder: Directory containing image files.
        suffix: Glob pattern used to select files.

    Returns:
        Sorted list of matching image paths.

    Raises:
        RuntimeError: If no matching files are found.
    """
    files = sorted(folder.glob(suffix))
    if not files:
        raise RuntimeError(f"No files found in {folder}")
    return files


def build_depth_index(depth_dir: Path) -> tuple[np.ndarray, list[Path]]:
    """
    Build a timestamp index for a TUM RGB-D depth sequence.

    Args:
        depth_dir: Directory containing depth images.

    Returns:
        Tuple containing:
        - depth_ts: NumPy array of depth timestamps,
        - depth_files: Sorted list of depth image paths.
    """
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
    """
    Find the nearest depth frame to a given RGB frame timestamp.

    Inspiration:
    - Nearest-neighbour timestamp association is used here so RGB and depth
      frames can be paired within a controlled temporal tolerance.

    Args:
        rgb_path: Path to the RGB image.
        depth_ts: Array of depth timestamps.
        depth_files: List of depth image paths aligned with depth_ts.
        max_dt: Maximum allowed timestamp difference.

    Returns:
        Path to the nearest acceptable depth frame, or None if no depth frame
        falls within the allowed time difference.
    """
    t = timestamp_from_path(rgb_path)
    k = int(np.argmin(np.abs(depth_ts - t)))
    if abs(depth_ts[k] - t) > max_dt:
        return None
    return depth_files[k]


def load_groundtruth(gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ground-truth timestamps and 3D positions from a TUM trajectory file.

    Args:
        gt_path: Path to the ground-truth text file.

    Returns:
        Tuple containing:
        - sorted ground-truth timestamps,
        - corresponding sorted XYZ positions.
    """
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
    """
    Interpolate ground-truth positions onto estimated trajectory timestamps.

    Inspiration:
    - Linear interpolation is used so benchmark reference positions can be
      compared directly with the estimated trajectory at matching times.

    Args:
        est_timestamps: Timestamps of the estimated trajectory.
        gt_timestamps: Ground-truth timestamps.
        gt_xyz: Ground-truth XYZ positions.

    Returns:
        Array of interpolated XYZ positions aligned to the
        estimated timestamps.
    """
    x = np.interp(est_timestamps, gt_timestamps, gt_xyz[:, 0])
    y = np.interp(est_timestamps, gt_timestamps, gt_xyz[:, 1])
    z = np.interp(est_timestamps, gt_timestamps, gt_xyz[:, 2])
    return np.column_stack([x, y, z])


def load_rgb_sequence(rgb_dir: Path) -> list[Path]:
    """
    Load a sorted RGB image sequence from a dataset directory.

    Args:
        rgb_dir: Directory containing RGB images.

    Returns:
        Sorted list of RGB image paths.
    """
    return list_image_files(rgb_dir, "*.png")
