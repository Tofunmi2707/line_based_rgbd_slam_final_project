from __future__ import annotations

"""
Centroid-based line front end for the baseline RGB-D SLAM pipeline.

This module implements the initial front-end variant used as
the baseline in the project. It performs:
- grayscale image loading,
- CLAHE preprocessing,
- LSD line detection,
- retention of the longest detected segments,
- simple geometric matching using line centroids, lengths, and orientations,
- histogram-based filtering of tentative matches,
- visualisation of detections and centroid matches.

Inspiration:
- CLAHE and LSD detection follow standard OpenCV image-processing workflows.
- The use of line centroids, segment length, and orientation as matching cues
  was a project-specific simplified baseline designed for debugging and
  comparison.
- The histogram-based filtering stage was implemented within the present
project to retain matches consistent with the dominant
displacement and angle trend of a frame pair.

Notes:
- This is a deliberately simplified baseline.
- Matched centroids are treated as point-like correspondences for later pose
  estimation, even though they do not represent full line geometry.
"""

from collections import defaultdict
from pathlib import Path
import math
import cv2
import numpy as np


def load_grayscale(path: str) -> np.ndarray:
    """
    Load an image in grayscale.

    Args:
        path: Path to the image file.

    Returns:
        Grayscale image.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalisation.

    Inspiration:
    - Standard OpenCV CLAHE preprocessing for local contrast enhancement.

    Args:
        img: Input grayscale image.
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid size.

    Returns:
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(img)


def detect_lines_lsd(img: np.ndarray) -> np.ndarray:
    """
    Detect line segments using OpenCV's Line Segment Detector.

    Inspiration:
    - Standard OpenCV LSD interface.

    Args:
        img: Input grayscale image.

    Returns:
        Array of detected line segments in OpenCV LSD format.
        Returns an empty array if no lines are detected.
    """
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]
    if lines is None:
        return np.empty((0, 1, 4), dtype=np.float32)
    return lines


def keep_top_k_by_length(lines: np.ndarray, k: int) -> np.ndarray:
    """
    Keep the k longest detected line segments.

    Args:
        lines: Input LSD line array.
        k: Maximum number of line segments to retain.

    Returns:
        Filtered line array containing the longest segments.
    """
    if len(lines) == 0:
        return lines

    xy = lines.reshape(-1, 4)
    lengths = np.hypot(xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1])
    idx = np.argsort(lengths)[::-1][:k]
    return lines[idx].reshape(-1, 1, 4)


def compute_features(lines: np.ndarray) -> tuple[np.ndarray,
                                                 np.ndarray,
                                                 np.ndarray]:
    """
    Compute centroid, length, and orientation for each line segment.

    Args:
        lines: Input LSD line array.

    Returns:
        Tuple containing:
        - mids: Line centroids with shape (N, 2),
        - lengths: Segment lengths in pixels,
        - angles: Segment orientations in degrees.
    """
    xy = lines.reshape(-1, 4)

    mids = np.stack(
        [
            (xy[:, 0] + xy[:, 2]) / 2.0,
            (xy[:, 1] + xy[:, 3]) / 2.0,
        ],
        axis=1,
    ).astype(np.float32)

    lengths = np.hypot(
        xy[:, 2] - xy[:, 0],
        xy[:, 3] - xy[:, 1],
    ).astype(np.float32)

    angles = np.degrees(
        np.arctan2(
            xy[:, 3] - xy[:, 1],
            xy[:, 2] - xy[:, 0],
        )
    ).astype(np.float32)

    return mids, lengths, angles


def build_grid(points: np.ndarray, grid_cell_size: int) -> dict:
    """
    Build a spatial grid index for 2D point lookup.

    Args:
        points: 2D points with shape (N, 2).
        grid_cell_size: Side length of each grid cell in pixels.

    Returns:
        Dictionary mapping grid-cell indices to point indices.
    """
    grid = defaultdict(list)
    for i, (x, y) in enumerate(points):
        key = (int(x) // grid_cell_size, int(y) // grid_cell_size)
        grid[key].append(i)
    return grid


def get_candidates(
    grid: dict,
    x: float,
    y: float,
    grid_cell_size: int,
) -> list[int]:
    """
    Retrieve candidate point indices from neighbouring grid cells.

    Args:
        grid: Spatial grid index.
        x: Query x coordinate.
        y: Query y coordinate.
        grid_cell_size: Grid-cell size in pixels.

    Returns:
        List of candidate indices from the surrounding 3 x 3 grid region.
    """
    gx = int(x) // grid_cell_size
    gy = int(y) // grid_cell_size
    candidates = []

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            candidates.extend(grid.get((gx + dx, gy + dy), []))

    return candidates


def fast_match(
    mids1: np.ndarray,
    lens1: np.ndarray,
    angs1: np.ndarray,
    mids2: np.ndarray,
    lens2: np.ndarray,
    angs2: np.ndarray,
    grid_cell_size: int,
    max_angle_diff_deg: float,
    max_centroid_dist_px: float,
    min_length_ratio: float,
    max_length_ratio: float,
    match_score_threshold: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Form tentative centroid-based matches using simple geometric cues.

    The score combines centroid distance, length difference, and angle
    difference. For each line in image 1, only the best candidate in image 2 is
    retained if it satisfies the threshold tests.

    Inspiration:
    - This is a project-specific simplified baseline matcher using centroid,
      length, and orientation instead of descriptor-based line matching.

    Args:
        mids1: Centroids from image 1.
        lens1: Segment lengths from image 1.
        angs1: Segment angles from image 1.
        mids2: Centroids from image 2.
        lens2: Segment lengths from image 2.
        angs2: Segment angles from image 2.
        grid_cell_size: Spatial lookup cell size in pixels.
        max_angle_diff_deg: Maximum allowed angle difference.
        max_centroid_dist_px: Maximum allowed centroid distance.
        min_length_ratio: Minimum accepted length ratio.
        max_length_ratio: Maximum accepted length ratio.
        match_score_threshold: Maximum accepted aggregate match score.

    Returns:
        List of tentative centroid matches as pairs (m1, m2).
    """
    grid = build_grid(mids2, grid_cell_size)
    matches = []

    for i in range(len(mids1)):
        m1 = mids1[i]
        L1 = lens1[i]
        A1 = angs1[i]

        candidate_indices = get_candidates(grid, m1[0], m1[1], grid_cell_size)

        best = None
        best_score = 1e9

        for j in candidate_indices:
            m2 = mids2[j]
            L2 = lens2[j]
            A2 = angs2[j]

            angle_diff = min(abs(A1 - A2), 360 - abs(A1 - A2))
            if angle_diff > max_angle_diff_deg:
                continue

            length_ratio = L1 / (L2 + 1e-6)
            if not (min_length_ratio < length_ratio < max_length_ratio):
                continue

            d = np.linalg.norm(m1 - m2)
            if d > max_centroid_dist_px:
                continue

            score = d + abs(L1 - L2) + 2.0 * angle_diff
            if score < best_score:
                best_score = score
                best = (m1, m2)

        if best is not None and best_score < match_score_threshold:
            matches.append(best)

    return matches


def histogram_filter(
    matches: list[tuple[np.ndarray, np.ndarray]],
    hist_bins: int,
    length_band_width: float,
    angle_band_width_deg: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict | None]:
    """
    Filter tentative matches using dominant connection length and angle bins.

    The method computes:
    - connection lengths between matched centroids,
    - connection angles between matched centroids,
    then retains matches that lie near the dominant histogram bins.

    Inspiration:
    - This is a project-specific filtering stage designed
      to reject matches that are inconsistent with the
      dominant displacement and angular trend.

    Args:
        matches: Tentative centroid matches.
        hist_bins: Number of histogram bins.
        length_band_width: Accepted band around the dominant connection length.
        angle_band_width_deg: Accepted band around the dominant
        connection angle.

    Returns:
        Tuple containing:
        - filtered matches,
        - diagnostics dictionary, or None if too few matches are available.
    """
    if len(matches) < 10:
        return [], None

    conn_lengths = np.array(
        [np.linalg.norm(m1 - m2) for m1, m2 in matches],
        dtype=float,
    )
    conn_angles = np.array(
        [
            math.degrees(math.atan2(m2[1] - m1[1], m2[0] - m1[0]))
            for m1, m2 in matches
        ],
        dtype=float,
    )

    L_hist, L_bins = np.histogram(conn_lengths, bins=hist_bins)
    A_hist, A_bins = np.histogram(conn_angles, bins=hist_bins)

    dominant_L = (
        L_bins[np.argmax(L_hist)] + L_bins[np.argmax(L_hist) + 1]
        ) / 2.0
    dominant_A = (
        A_bins[np.argmax(A_hist)] + A_bins[np.argmax(A_hist) + 1]
        ) / 2.0

    mask = (
        (np.abs(conn_lengths - dominant_L) < length_band_width)
        & (np.abs(conn_angles - dominant_A) < angle_band_width_deg)
    )

    filtered = [m for m, keep in zip(matches, mask) if keep]
    diagnostics = {
        "conn_lengths": conn_lengths,
        "conn_angles": conn_angles,
        "dominant_L": dominant_L,
        "dominant_A": dominant_A,
        "mask": mask,
    }
    return filtered, diagnostics


def draw_lsd_overlay(
    gray_img: np.ndarray,
    lines: np.ndarray,
    line_colour: tuple[int, int, int] = (0, 255, 255),
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw detected LSD line segments on a grayscale image.

    Args:
        gray_img: Input grayscale image.
        lines: LSD line array.
        line_colour: BGR line colour.
        line_thickness: Thickness of drawn lines.

    Returns:
        Colour visualisation image.
    """
    vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, line)
        cv2.line(
            vis,
            (x1, y1),
            (x2, y2),
            line_colour,
            line_thickness,
            cv2.LINE_AA,
        )
    return vis


def draw_centroid_matches(
    gray1: np.ndarray,
    gray2: np.ndarray,
    matches: list[tuple[np.ndarray, np.ndarray]],
    max_draw: int | None = 20,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw centroid correspondences between two grayscale images.

    Args:
        gray1: First grayscale image.
        gray2: Second grayscale image.
        matches: List of centroid match pairs.
        max_draw: Maximum number of matches to draw, or None to draw all.
        line_thickness: Thickness of drawn correspondence lines.

    Returns:
        Side-by-side visualisation image.
    """
    img1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    W = w1 + w2

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    draw_matches = matches if max_draw is None else matches[:max_draw]

    colours = [
        (255, 0, 0),
        (0, 180, 255),
        (0, 200, 0),
        (180, 0, 255),
        (255, 120, 0),
        (0, 220, 220),
    ]

    for i, (m1, m2) in enumerate(draw_matches):
        colour = colours[i % len(colours)]
        p1 = (int(round(m1[0])), int(round(m1[1])))
        p2 = (int(round(m2[0] + w1)), int(round(m2[1])))

        cv2.line(canvas, p1, p2, colour, line_thickness, cv2.LINE_AA)
        cv2.circle(canvas, p1, 5, colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 5, colour, -1, cv2.LINE_AA)

    return canvas


def save_frontend_visuals(
    img1_path: str,
    img2_path: str,
    cfg,
    output_dir: str | Path,
) -> dict | None:
    """
    Save the main visual outputs for the centroid-based front end.

    Saved outputs include:
    - CLAHE before/after comparison,
    - LSD overlay,
    - raw centroid matches,
    - filtered centroid matches.

    Args:
        img1_path: Path to first image.
        img2_path: Path to second image.
        cfg: Front-end configuration object.
        output_dir: Directory for saved visual outputs.

    Returns:
        Front-end result dictionary, or None if processing fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = process_frame_pair_frontend(img1_path, img2_path, cfg)
    if result is None:
        return None

    clahe_vis = np.hstack([result["img1_raw"], result["img1"]])
    cv2.imwrite(str(output_dir / "clahe_before_after.png"), clahe_vis)

    lsd_overlay = draw_lsd_overlay(result["img1"], result["lines1"])
    cv2.imwrite(str(output_dir / "lsd_overlay.png"), lsd_overlay)

    raw_vis = draw_centroid_matches(
        result["img1"],
        result["img2"],
        result["raw_matches"],
        max_draw=20,
    )
    cv2.imwrite(str(output_dir / "raw_matches.png"), raw_vis)

    filtered_vis = draw_centroid_matches(
        result["img1"],
        result["img2"],
        result["filtered_matches"],
        max_draw=20,
    )
    cv2.imwrite(str(output_dir / "filtered_matches.png"), filtered_vis)

    return result


def process_frame_pair_frontend(
    img1_path: str,
    img2_path: str,
    cfg,
) -> dict | None:
    """
    Run the centroid-based front end on a single frame pair.

    The function:
    1. loads grayscale images,
    2. applies CLAHE,
    3. detects LSD line segments,
    4. keeps the longest lines,
    5. computes centroid-based line features,
    6. forms tentative matches,
    7. applies histogram-based filtering,
    8. returns filtered centroid correspondences for pose estimation.

    Args:
        img1_path: Path to first image.
        img2_path: Path to second image.
        cfg: Configuration object containing front-end parameters.

    Returns:
        Dictionary containing images, line detections, match lists, centroid
        correspondence arrays, and histogram diagnostics, or None if processing
        fails due to insufficient support.
    """
    img1_raw = load_grayscale(img1_path)
    img2_raw = load_grayscale(img2_path)

    img1 = apply_clahe(img1_raw)
    img2 = apply_clahe(img2_raw)

    lines1 = keep_top_k_by_length(detect_lines_lsd(img1), cfg.max_lines)
    lines2 = keep_top_k_by_length(detect_lines_lsd(img2), cfg.max_lines)

    if len(lines1) < 10 or len(lines2) < 10:
        return None

    mids1, lens1, angs1 = compute_features(lines1)
    mids2, lens2, angs2 = compute_features(lines2)

    raw_matches = fast_match(
        mids1,
        lens1,
        angs1,
        mids2,
        lens2,
        angs2,
        cfg.grid_cell_size,
        cfg.max_angle_diff_deg,
        cfg.max_centroid_dist_px,
        cfg.min_length_ratio,
        cfg.max_length_ratio,
        cfg.match_score_threshold,
    )

    filtered_matches, hist = histogram_filter(
        raw_matches,
        cfg.hist_bins,
        cfg.length_band_width,
        cfg.angle_band_width_deg,
    )

    if len(filtered_matches) < cfg.min_filtered_matches:
        return None

    A = np.array([m1 for (m1, m2) in filtered_matches], dtype=np.float32)
    B = np.array([m2 for (m1, m2) in filtered_matches], dtype=np.float32)

    return {
        "img1_raw": img1_raw,
        "img2_raw": img2_raw,
        "img1": img1,
        "img2": img2,
        "lines1": lines1,
        "lines2": lines2,
        "raw_matches": raw_matches,
        "filtered_matches": filtered_matches,
        "A": A,
        "B": B,
        "hist": hist,
    }
