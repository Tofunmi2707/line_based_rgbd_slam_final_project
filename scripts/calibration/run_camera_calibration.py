from __future__ import annotations

# Inspiration: OpenCV checkerboard calibration workflow (corner detection,
# calibrateCamera, undistortion, reprojection analysis), with project-specific
# output organisation, figure export, and report-ready summary files.

from pathlib import Path
import glob
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "scripts" / "calibration" / "images"
OUTPUT_DIR = PROJECT_ROOT / "results" / "calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKERBOARD = (9, 6)  # inner corners: (columns, rows)
USE_SB_DETECTOR = True
SHOW_WINDOWS = False

CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)


def draw_clean_chessboard_corners(
    image: np.ndarray,
    corners: np.ndarray,
    pattern_size: tuple[int, int],
    circle_radius: int = 8,
    line_thickness: int = 4,
) -> np.ndarray:
    # Inspiration: OpenCV drawing primitives; colour-coded row/column overlay
    # implemented here for cleaner report figures.
    img = image.copy()

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cols, rows = pattern_size
    pts = corners.reshape(rows, cols, 2)

    for r in range(rows):
        for c in range(cols - 1):
            p1 = tuple(np.round(pts[r, c]).astype(int))
            p2 = tuple(np.round(pts[r, c + 1]).astype(int))
            cv2.line(img, p1, p2, (0, 255, 255), line_thickness, cv2.LINE_AA)

    for c in range(cols):
        for r in range(rows - 1):
            p1 = tuple(np.round(pts[r, c]).astype(int))
            p2 = tuple(np.round(pts[r + 1, c]).astype(int))
            cv2.line(img, p1, p2, (255, 0, 255), line_thickness, cv2.LINE_AA)

    for r in range(rows):
        for c in range(cols):
            x, y = np.round(pts[r, c]).astype(int)
            cv2.circle(img, (x, y), circle_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), circle_radius, (255, 255, 255), -1, cv2.LINE_AA)

    return img


def detect_chessboard(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None, str]:
    # Inspiration: OpenCV findChessboardCornersSB and findChessboardCorners;
    # fallback logic added here so calibration is more robust across image sets.
    if USE_SB_DETECTOR and hasattr(cv2, "findChessboardCornersSB"):
        ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
        if ret:
            return True, corners, "findChessboardCornersSB"

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        return True, corners, "findChessboardCorners"

    return False, None, "none"


def save_undistortion_figure(
    original_bgr: np.ndarray,
    undistorted_bgr: np.ndarray,
    save_path: Path,
) -> None:
    # Inspiration: matplotlib figure export; side-by-side report figure layout
    # implemented here for dissertation-ready output.
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6), facecolor="white")

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original image", fontsize=18)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(undistorted_rgb)
    plt.title("Undistorted image", fontsize=18)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_reprojection_plot(errors: list[float], save_path: Path) -> None:
    # Inspiration: standard matplotlib bar chart; mean-error overlay added here
    # for clearer dissertation presentation.
    plt.figure(figsize=(10, 5), facecolor="white")
    ax = plt.gca()

    mean_error = float(np.mean(errors))

    ax.bar(
        range(1, len(errors) + 1),
        errors,
        color="#355C7D",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axhline(
        mean_error,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_error:.3f} px",
    )

    ax.set_title("Per-image reprojection error", fontsize=20, pad=14)
    ax.set_xlabel("Calibration image index", fontsize=15)
    ax.set_ylabel("Reprojection error (pixels)", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_calibration_summary_txt(
    save_path: Path,
    checkerboard: tuple[int, int],
    total_images: int,
    successful_detections: int,
    detector_names: set[str],
    mean_error: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    # Inspiration: examiner-facing plain-text summary for auditability.
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Calibration summary\n")
        f.write("-------------------\n")
        f.write(f"Checkerboard: {checkerboard}\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"Successful detections: {successful_detections}\n")
        f.write(f"Detection methods used: {sorted(list(detector_names))}\n")
        f.write(f"Mean reprojection error: {mean_error:.6f}\n\n")
        f.write("Camera matrix:\n")
        f.write(str(camera_matrix))
        f.write("\n\nDistortion coefficients:\n")
        f.write(str(dist_coeffs))
        f.write("\n")


def main() -> None:
    # Inspiration: Zhang-style planar calibration workflow through OpenCV,
    # adapted here to save all outputs in a dedicated report-results folder.
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(str(IMAGE_DIR / ext)))
    images = sorted(images)

    if not images:
        raise FileNotFoundError(f"No calibration images found in: {IMAGE_DIR}")

    print(f"Found {len(images)} calibration images")
    print(f"Using checkerboard pattern: {CHECKERBOARD}")

    successful_images = []
    detector_names_used = set()
    image_size = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Could not read image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        ret, corners, method_name = detect_chessboard(gray, CHECKERBOARD)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            successful_images.append(fname)
            detector_names_used.add(method_name)

            bold_img = draw_clean_chessboard_corners(img, corners, CHECKERBOARD)
            base = Path(fname).stem
            cv2.imwrite(str(OUTPUT_DIR / f"{base}_corners_bold.jpg"), bold_img)

            print(f"Detected corners in {Path(fname).name} using {method_name}")

            if SHOW_WINDOWS:
                cv2.imshow("Detected corners", bold_img)
                cv2.waitKey(250)
        else:
            print(f"Chessboard not detected in: {Path(fname).name}")

    cv2.destroyAllWindows()

    if not objpoints or image_size is None:
        raise RuntimeError("No valid chessboard detections found.")

    flags = cv2.CALIB_FIX_K3
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )

    print("Calibration RMS from OpenCV:", ret)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    per_image_errors = []
    for i in range(len(objpoints)):
        projected_points, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )
        error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
        per_image_errors.append(float(error))

    mean_error = float(np.mean(per_image_errors))
    print("Mean reprojection error:", mean_error)

    sample_img = cv2.imread(successful_images[0])
    h, w = sample_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(sample_img, mtx, dist, None, newcameramtx)

    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted_cropped = undistorted[y:y + h_roi, x:x + w_roi]
    else:
        undistorted_cropped = undistorted.copy()

    cv2.imwrite(str(OUTPUT_DIR / "original_sample.jpg"), sample_img)
    cv2.imwrite(str(OUTPUT_DIR / "undistorted_sample_full.jpg"), undistorted)
    cv2.imwrite(str(OUTPUT_DIR / "undistorted_sample_cropped.jpg"), undistorted_cropped)

    save_undistortion_figure(
        sample_img,
        undistorted_cropped,
        OUTPUT_DIR / "before_after_undistortion.png",
    )

    save_reprojection_plot(
        per_image_errors,
        OUTPUT_DIR / "reprojection_error_plot.png",
    )

    with open(OUTPUT_DIR / "camera_params.pkl", "wb") as f:
        pickle.dump({"mtx": mtx, "dist": dist}, f)

    save_calibration_summary_txt(
        OUTPUT_DIR / "calibration_summary.txt",
        CHECKERBOARD,
        len(images),
        len(objpoints),
        detector_names_used,
        mean_error,
        mtx,
        dist,
    )

    print(f"Saved calibration outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
