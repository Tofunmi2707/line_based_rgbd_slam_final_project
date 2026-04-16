from __future__ import annotations

"""
Pose-estimation utilities for the line-based RGB-D SLAM pipeline.

This module provides:
- depth-image loading,
- pixel back-projection into 3D camera coordinates,
- calibrated two-view pose estimation using the Essential matrix,
- depth-assisted metric translation estimation,
- rigid-transform construction.

Inspiration:
- The calibrated two-view workflow follows the standard OpenCV pipeline based on
  `findEssentialMat` and `recoverPose`.
- Pixel back-projection follows the standard pinhole RGB-D camera model used in
  visual odometry and RGB-D reconstruction.
- The metric translation stage is a project-specific integration step that uses
  valid depth-backed correspondences after rotation recovery.
"""

import cv2
import numpy as np


def load_depth_image(depth_path: str | None) -> np.ndarray | None:
    """
    Load a depth image from disk as a floating-point array.

    Args:
        depth_path: Path to the depth image file, or None.

    Returns:
        Depth image as a float32 NumPy array, or None if loading fails.
    """
    if depth_path is None:
        return None

    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None

    return d.astype(np.float32)


def backproject_pixels_to_3d(
    uv: np.ndarray,
    depth_img: np.ndarray | None,
    K: np.ndarray,
    depth_scale: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Back-project image pixels into 3D camera-frame coordinates.

    Inspiration:
    - Standard RGB-D pinhole back-projection:
      X = (u - cx) Z / fx, Y = (v - cy) Z / fy, Z = d / depth_scale.

    Args:
        uv: Array of image coordinates with shape (N, 2).
        depth_img: Depth image corresponding to the frame.
        K: 3 x 3 camera intrinsic matrix.
        depth_scale: Depth scaling factor used by the dataset.

    Returns:
        Tuple containing:
        - P: Array of 3D points with shape (N, 3), or None if no valid depth exists.
        - valid: Boolean mask indicating which points have valid positive depth,
          or None if back-projection fails.
    """
    if depth_img is None:
        return None, None

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    H, W = depth_img.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]

    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)

    d_raw = depth_img[vi, ui]
    Z = d_raw / float(depth_scale)

    valid = Z > 0.0
    if valid.sum() == 0:
        return None, None

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    P = np.column_stack([X, Y, Z]).astype(np.float32)
    return P, valid


def estimate_pose_essential(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Estimate relative pose from calibrated two-view correspondences.

    Inspiration:
    - OpenCV calibrated two-view workflow using `findEssentialMat` followed by
      `recoverPose`.

    Args:
        A: Image points from frame 1 with shape (N, 2).
        B: Corresponding image points from frame 2 with shape (N, 2).
        K: 3 x 3 camera intrinsic matrix.

    Returns:
        Tuple containing:
        - R: Recovered 3 x 3 rotation matrix,
        - t: Recovered 3 x 1 translation direction,
        - inlier_mask: Boolean mask of Essential-matrix inliers.

        Returns None if Essential-matrix estimation fails.
    """
    E, mask = cv2.findEssentialMat(
        A,
        B,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    if E is None or mask is None:
        return None

    _, R, t, mask_pose = cv2.recoverPose(E, A, B, K, mask=mask)
    inlier_mask = mask_pose.ravel().astype(bool)
    return R, t, inlier_mask


def estimate_metric_translation(
    A: np.ndarray,
    B: np.ndarray,
    depth1: np.ndarray,
    depth2: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
    R: np.ndarray,
) -> tuple[np.ndarray | None, int]:
    """
    Estimate metric translation using valid depth-backed correspondences.

    The function back-projects the inlier image correspondences from both frames
    into 3D, applies the recovered rotation to the first-frame points, and
    estimates translation from the mean displacement between the rotated and
    observed 3D points.

    Inspiration:
    - Standard RGB-D back-projection is used here.
    - The final translation computation is a project-specific integration choice
      for linking calibrated two-view pose with available depth measurements.

    Args:
        A: Inlier image points from frame 1.
        B: Inlier image points from frame 2.
        depth1: Depth image for frame 1.
        depth2: Depth image for frame 2.
        K: 3 x 3 camera intrinsic matrix.
        depth_scale: Depth scaling factor.
        R: Recovered relative rotation matrix.

    Returns:
        Tuple containing:
        - t_metric: Estimated 3 x 1 metric translation vector, or None if
          insufficient valid 3D correspondences are available.
        - num_metric_points: Number of valid depth-backed correspondences used.
    """
    P1, v1 = backproject_pixels_to_3d(A, depth1, K, depth_scale)
    P2, v2 = backproject_pixels_to_3d(B, depth2, K, depth_scale)
    if P1 is None or P2 is None:
        return None, 0

    valid3d = v1 & v2
    if int(valid3d.sum()) < 10:
        return None, int(valid3d.sum())

    P1 = P1[valid3d]
    P2 = P2[valid3d]

    RP1 = (R @ P1.T).T
    t_metric = (P2 - RP1).mean(axis=0).reshape(3, 1).astype(np.float32)
    return t_metric, int(valid3d.sum())


def process_frame_pair_pose(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    depth1_path: str | None,
    depth2_path: str | None,
    depth_scale: float,
    min_essential_inliers: int,
    min_metric_points: int,
) -> tuple[dict | None, dict]:
    """
    Estimate pose for a single frame pair and return debug information.

    The function:
    1. estimates calibrated two-view pose using the Essential matrix,
    2. checks the number of Essential-matrix inliers,
    3. loads the corresponding depth images,
    4. attempts metric translation recovery,
    5. records detailed rejection reasons for later diagnosis.

    Inspiration:
    - The calibrated two-view stage follows the standard OpenCV Essential-matrix
      workflow.
    - The structured debug dictionary and explicit rejection categories are
      project-specific additions used to support the odometry analysis chapter.

    Args:
        A: Matched image points from frame 1.
        B: Matched image points from frame 2.
        K: 3 x 3 camera intrinsic matrix.
        depth1_path: Path to depth image for frame 1.
        depth2_path: Path to depth image for frame 2.
        depth_scale: Depth scaling factor.
        min_essential_inliers: Minimum number of Essential-matrix inliers required.
        min_metric_points: Minimum number of valid 3D correspondences required.

    Returns:
        Tuple containing:
        - pose: Dictionary with pose outputs if estimation succeeds sufficiently,
          or None if the Essential-matrix stage fails outright.
        - debug: Dictionary containing intermediate counts, flags, norms, and
          rejection reasons.
    """
    debug = {
        "num_input_matches": int(len(A)),
        "essential_success": False,
        "num_essential_inliers": 0,
        "depth1_loaded": False,
        "depth2_loaded": False,
        "num_metric_points": 0,
        "metric": False,
        "reject_reason": None,
        "t_direction_norm": None,
        "t_metric_norm": None,
    }

    out = estimate_pose_essential(A, B, K)
    if out is None:
        debug["reject_reason"] = "essential_failed"
        return None, debug

    R, t_dir, inlier_mask = out
    num_inliers_2d = int(inlier_mask.sum())

    debug["essential_success"] = True
    debug["num_essential_inliers"] = num_inliers_2d
    debug["t_direction_norm"] = float(np.linalg.norm(t_dir.reshape(3)))

    if num_inliers_2d < min_essential_inliers:
        debug["reject_reason"] = "too_few_essential_inliers"
        return None, debug

    depth1 = load_depth_image(depth1_path)
    depth2 = load_depth_image(depth2_path)

    debug["depth1_loaded"] = depth1 is not None
    debug["depth2_loaded"] = depth2 is not None

    if depth1 is None or depth2 is None:
        debug["reject_reason"] = "depth_load_failed"
        return {
            "R": R,
            "t": t_dir,
            "num_inliers": num_inliers_2d,
            "metric": False,
        }, debug

    A_in = A[inlier_mask]
    B_in = B[inlier_mask]

    t_metric, num_metric = estimate_metric_translation(
        A_in,
        B_in,
        depth1,
        depth2,
        K,
        depth_scale,
        R,
    )

    debug["num_metric_points"] = int(num_metric)

    if t_metric is None or num_metric < min_metric_points:
        debug["reject_reason"] = "too_few_metric_points"
        return {
            "R": R,
            "t": t_dir,
            "num_inliers": num_inliers_2d,
            "metric": False,
        }, debug

    debug["metric"] = True
    debug["t_metric_norm"] = float(np.linalg.norm(t_metric.reshape(3)))
    debug["reject_reason"] = None

    return {
        "R": R,
        "t": t_metric,
        "num_inliers": num_metric,
        "metric": True,
    }, debug


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Construct a 4 x 4 homogeneous rigid-body transform from rotation and translation.

    Args:
        R: 3 x 3 rotation matrix.
        t: 3 x 1 or length-3 translation vector.

    Returns:
        4 x 4 homogeneous transformation matrix.
    """
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T
