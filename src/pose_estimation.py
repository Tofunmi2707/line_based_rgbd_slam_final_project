import cv2
import numpy as np


def load_depth_image(depth_path: str) -> np.ndarray | None:
    if depth_path is None:
        return None
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return d.astype(np.float32)


def backproject_pixels_to_3d(
    uv: np.ndarray,
    depth_img: np.ndarray,
    K: np.ndarray,
    depth_scale: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def estimate_pose_essential(A: np.ndarray, B: np.ndarray, K: np.ndarray):
    E, mask = cv2.findEssentialMat(
        A, B, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
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
):
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
):
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
        A_in, B_in, depth1, depth2, K, depth_scale, R
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
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T
