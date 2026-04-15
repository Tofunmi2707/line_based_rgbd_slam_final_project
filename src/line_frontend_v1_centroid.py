from collections import defaultdict
import math
import cv2
import numpy as np
from pathlib import Path


def load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    return clahe.apply(img)


def detect_lines_lsd(img: np.ndarray) -> np.ndarray:
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]
    if lines is None:
        return np.empty((0, 1, 4), dtype=np.float32)
    return lines


def keep_top_k_by_length(lines: np.ndarray, k: int) -> np.ndarray:
    if len(lines) == 0:
        return lines
    xy = lines.reshape(-1, 4)
    lengths = np.hypot(xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1])
    idx = np.argsort(lengths)[::-1][:k]
    return lines[idx].reshape(-1, 1, 4)


def compute_features(lines: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = lines.reshape(-1, 4)

    mids = np.stack([
        (xy[:, 0] + xy[:, 2]) / 2.0,
        (xy[:, 1] + xy[:, 3]) / 2.0
    ], axis=1).astype(np.float32)

    lengths = np.hypot(xy[:, 2] - xy[:, 0],
                       xy[:, 3] - xy[:, 1]).astype(np.float32)

    angles = np.degrees(np.arctan2(
        xy[:, 3] - xy[:, 1],
        xy[:, 2] - xy[:, 0]
    )).astype(np.float32)

    return mids, lengths, angles


def build_grid(points: np.ndarray, grid_cell_size: int) -> dict:
    grid = defaultdict(list)
    for i, (x, y) in enumerate(points):
        key = (int(x) // grid_cell_size, int(y) // grid_cell_size)
        grid[key].append(i)
    return grid


def get_candidates(grid: dict,
                   x: float,
                   y: float,
                   grid_cell_size: int) -> list[int]:
    gx = int(x) // grid_cell_size
    gy = int(y) // grid_cell_size
    candidates = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            candidates.extend(grid.get((gx + dx, gy + dy), []))
    return candidates


def fast_match(
    mids1, lens1, angs1,
    mids2, lens2, angs2,
    grid_cell_size: int,
    max_angle_diff_deg: float,
    max_centroid_dist_px: float,
    min_length_ratio: float,
    max_length_ratio: float,
    match_score_threshold: float,
):
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

            score = d + abs(L1 - L2) + 2 * angle_diff
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
):
    if len(matches) < 10:
        return [], None

    conn_lengths = np.array([np.linalg.norm(m1 - m2) for m1, m2 in matches])
    conn_angles = np.array([
        math.degrees(math.atan2(m2[1] - m1[1], m2[0] - m1[0]))
        for m1, m2 in matches
    ])

    L_hist, L_bins = np.histogram(conn_lengths, bins=hist_bins)
    A_hist, A_bins = np.histogram(conn_angles, bins=hist_bins)

    dominant_L = (L_bins[np.argmax(L_hist)] +
                  L_bins[np.argmax(L_hist) + 1]) / 2.0
    dominant_A = (A_bins[np.argmax(A_hist)] +
                  A_bins[np.argmax(A_hist) + 1]) / 2.0

    mask = (
        (np.abs(conn_lengths - dominant_L) < length_band_width) &
        (np.abs(conn_angles - dominant_A) < angle_band_width_deg)
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


def draw_lsd_overlay(gray_img: np.ndarray,
                     lines: np.ndarray,
                     line_colour=(0, 255, 255),
                     line_thickness=2) -> np.ndarray:
    vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, line)
        cv2.line(vis,
                 (x1, y1),
                 (x2, y2),
                 line_colour,
                 line_thickness,
                 cv2.LINE_AA)
    return vis


def draw_centroid_matches(gray1: np.ndarray,
                          gray2: np.ndarray,
                          matches: list[tuple[np.ndarray, np.ndarray]],
                          max_draw: int | None = 20,
                          line_thickness: int = 2) -> np.ndarray:
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


def save_frontend_visuals(img1_path: str,
                          img2_path: str,
                          cfg,
                          output_dir: str | Path) -> dict | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = process_frame_pair_frontend(img1_path, img2_path, cfg)
    if result is None:
        return None

    clahe_vis = np.hstack([result["img1_raw"], result["img1"]])
    cv2.imwrite(str(output_dir / "clahe_before_after.png"), clahe_vis)

    lsd_overlay = draw_lsd_overlay(result["img1"], result["lines1"])
    cv2.imwrite(str(output_dir / "lsd_overlay.png"), lsd_overlay)

    raw_vis = draw_centroid_matches(result["img1"],
                                    result["img2"],
                                    result["raw_matches"],
                                    max_draw=20)
    cv2.imwrite(str(output_dir / "raw_matches.png"), raw_vis)

    filtered_vis = draw_centroid_matches(result["img1"],
                                         result["img2"],
                                         result["filtered_matches"],
                                         max_draw=20)
    cv2.imwrite(str(output_dir / "filtered_matches.png"), filtered_vis)

    return result


def process_frame_pair_frontend(img1_path: str,
                                img2_path: str,
                                cfg) -> dict | None:
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
        mids1, lens1, angs1,
        mids2, lens2, angs2,
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
        cfg.angle_band_width_deg
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
