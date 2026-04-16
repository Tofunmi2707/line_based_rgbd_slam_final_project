from __future__ import annotations

"""
LBD-based line front end with endpoint correspondences.

This module implements the second front-end variant used in the project. It
performs:
- grayscale image loading,
- CLAHE preprocessing,
- LSD line detection,
- retention of the longest detected segments,
- conversion of detected segments into OpenCV KeyLine objects,
- LBD-style descriptor computation using OpenCV's line_descriptor module,
- descriptor matching with Lowe-style ratio filtering,
- one-to-one train-side match enforcement,
- conversion of matched line segments into endpoint point correspondences,
- visualisation of detections and tentative line matches.

Inspiration:
- CLAHE and LSD detection follow standard OpenCV image-processing workflows.
- Line description and matching follow the OpenCV line_descriptor workflow,
  which implements binary descriptors over KeyLine objects and supports
  descriptor-based line matching.
- The endpoint-correspondence construction,
  endpoint-order check, and the use of
  matched endpoints as pose-estimation input were integrated within the present
  project as a more geometrically meaningful alternative to centroid-based
  correspondences.

Notes:
- This variant keeps descriptor matching but does not yet apply the later
  geometric consistency filtering stage used in V3.
- Matched line segments are converted into endpoint point
  correspondences for the downstream Essential-matrix
  pose-estimation stage.
"""

from pathlib import Path
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


def detect_lines_lsd(img: np.ndarray, max_lines: int) -> np.ndarray:
    """
    Detect line segments using OpenCV's Line Segment Detector and keep the
    longest detections.

    Inspiration:
    - Standard OpenCV LSD interface.

    Args:
        img: Input grayscale image.
        max_lines: Maximum number of detected lines to retain.

    Returns:
        Array of retained line segments in OpenCV LSD format.
        Returns an empty array if no lines are detected.
    """
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]
    if lines is None:
        return np.empty((0, 1, 4), dtype=np.float32)

    xy = lines.reshape(-1, 4)
    lengths = np.hypot(xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1])
    idx = np.argsort(lengths)[::-1][:max_lines]
    return lines[idx].reshape(-1, 1, 4)


def lines_to_keylines(lines: np.ndarray) -> list:
    """
    Convert LSD line segments into OpenCV KeyLine objects.

    Inspiration:
    - OpenCV line_descriptor uses KeyLine objects as the line primitive for
      descriptor computation.

    Args:
        lines: Array of detected line segments in LSD format.

    Returns:
        List of KeyLine objects.
    """
    keylines = []

    for i, line in enumerate(lines.reshape(-1, 4)):
        x1, y1, x2, y2 = map(float, line)

        kl = cv2.line_descriptor.KeyLine()
        kl.startPointX = x1
        kl.startPointY = y1
        kl.endPointX = x2
        kl.endPointY = y2
        kl.sPointInOctaveX = x1
        kl.sPointInOctaveY = y1
        kl.ePointInOctaveX = x2
        kl.ePointInOctaveY = y2
        kl.lineLength = float(np.hypot(x2 - x1, y2 - y1))
        kl.angle = float(np.arctan2(y2 - y1, x2 - x1))
        kl.class_id = i
        kl.octave = 0
        kl.pt = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        kl.response = kl.lineLength
        kl.size = kl.lineLength
        kl.numOfPixels = int(max(2, round(kl.lineLength)))
        keylines.append(kl)

    return keylines


def compute_lbd_descriptors(img: np.ndarray,
                            lines: np.ndarray) -> tuple[list,
                                                        np.ndarray | None]:
    """
    Compute binary line descriptors for detected line segments.

    Inspiration:
    - OpenCV BinaryDescriptor over KeyLine objects, corresponding to an
      LBD-style line-descriptor workflow.

    Args:
        img: Input grayscale image.
        lines: Detected line segments.

    Returns:
        Tuple containing:
        - list of KeyLine objects,
        - descriptor array, or None if descriptor computation fails.
    """
    keylines = lines_to_keylines(lines)

    bd = cv2.line_descriptor.BinaryDescriptor.createBinaryDescriptor()
    keylines, desc = bd.compute(img, keylines)

    if desc is None or len(keylines) == 0:
        return [], None

    return keylines, desc


def endpoint_pair_from_match(kl1, kl2) -> tuple[np.ndarray, np.ndarray]:
    """
    Build endpoint pairs for a matched line pair and enforce consistent
    endpoint ordering.

    The function compares same-order and swapped-order endpoint distances, then
    reorders the second line if needed so that corresponding endpoints are
    paired more consistently.

    Inspiration:
    - This endpoint-order check is a project-specific step used so matched line
      segments produce more stable point correspondences for downstream pose
      estimation.

    Args:
        kl1: KeyLine from image 1.
        kl2: Matching KeyLine from image 2.

    Returns:
        Two arrays of shape (2, 2), giving ordered endpoints from image 1 and
        image 2.
    """
    A = np.array(
        [
            [kl1.startPointX, kl1.startPointY],
            [kl1.endPointX, kl1.endPointY],
        ],
        dtype=np.float32,
    )

    B = np.array(
        [
            [kl2.startPointX, kl2.startPointY],
            [kl2.endPointX, kl2.endPointY],
        ],
        dtype=np.float32,
    )

    d_same = np.linalg.norm(A[0] - B[0]) + np.linalg.norm(A[1] - B[1])
    d_swap = np.linalg.norm(A[0] - B[1]) + np.linalg.norm(A[1] - B[0])

    if d_swap < d_same:
        B = B[::-1]

    return A, B


def build_endpoint_arrays(matches,
                          keylines1,
                          keylines2) -> tuple[np.ndarray,
                                              np.ndarray,
                                              list[tuple]]:
    """
    Convert matched KeyLine pairs into endpoint point-correspondence arrays.

    Each accepted line match contributes two point correspondences,
    one for each ordered endpoint.

    Args:
        matches: Descriptor matches.
        keylines1: KeyLines from image 1.
        keylines2: KeyLines from image 2.

    Returns:
        Tuple containing:
        - A: Endpoint array from image 1 with shape (2M, 2),
        - B: Endpoint array from image 2 with shape (2M, 2),
        - matched_pairs: List of matched KeyLine pairs.
    """
    A_pts = []
    B_pts = []
    matched_pairs = []

    for m in matches:
        kl1 = keylines1[m.queryIdx]
        kl2 = keylines2[m.trainIdx]

        A_end, B_end = endpoint_pair_from_match(kl1, kl2)
        A_pts.extend(A_end)
        B_pts.extend(B_end)
        matched_pairs.append((kl1, kl2))

    A = np.asarray(A_pts, dtype=np.float32)
    B = np.asarray(B_pts, dtype=np.float32)
    return A, B, matched_pairs


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


def draw_line_matches(
    gray1: np.ndarray,
    gray2: np.ndarray,
    keylines1,
    keylines2,
    matches,
    max_draw: int | None = 12,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw matched line centroids between two images for visual inspection.

    Args:
        gray1: First grayscale image.
        gray2: Second grayscale image.
        keylines1: KeyLines from image 1.
        keylines2: KeyLines from image 2.
        matches: Descriptor matches.
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

    draw_matches = sorted(matches, key=lambda m: m.distance)
    draw_matches = draw_matches if max_draw is None else draw_matches[:max_draw]  # noqa: E501

    colours = [
        (255, 0, 0),
        (0, 180, 255),
        (0, 200, 0),
        (180, 0, 255),
        (255, 120, 0),
        (0, 220, 220),
    ]

    for i, m in enumerate(draw_matches):
        kl1 = keylines1[m.queryIdx]
        kl2 = keylines2[m.trainIdx]
        colour = colours[i % len(colours)]

        p1 = (int(round(kl1.pt[0])), int(round(kl1.pt[1])))
        p2 = (int(round(kl2.pt[0] + w1)), int(round(kl2.pt[1])))

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
    Save the main visual outputs for the LBD-endpoint front end.

    Saved outputs include:
    - CLAHE before/after comparison,
    - LSD overlay,
    - raw descriptor-match visualisation.

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

    raw_match_vis = draw_line_matches(
        result["img1"],
        result["img2"],
        result["keylines1"],
        result["keylines2"],
        result["raw_matches"],
        max_draw=20,
    )
    cv2.imwrite(str(output_dir / "raw_matches.png"), raw_match_vis)

    return result


def process_frame_pair_frontend(
    img1_path: str,
    img2_path: str,
    cfg,
) -> dict | None:
    """
    Run the LBD-endpoint front end on a single frame pair.

    The function:
    1. loads grayscale images,
    2. applies CLAHE,
    3. detects LSD line segments,
    4. computes binary line descriptors,
    5. performs descriptor matching with ratio filtering,
    6. enforces one-to-one use of train-side matches,
    7. converts matched lines into endpoint point correspondences.

    Args:
        img1_path: Path to first image.
        img2_path: Path to second image.
        cfg: Configuration object containing front-end parameters.

    Returns:
        Dictionary containing images, detections, KeyLines, match lists, and
        endpoint correspondence arrays, or None if processing fails due to
        insufficient support.
    """
    img1_raw = load_grayscale(img1_path)
    img2_raw = load_grayscale(img2_path)

    img1 = apply_clahe(img1_raw)
    img2 = apply_clahe(img2_raw)

    lines1 = detect_lines_lsd(img1, cfg.max_lines)
    lines2 = detect_lines_lsd(img2, cfg.max_lines)

    if len(lines1) < 10 or len(lines2) < 10:
        return None

    keylines1, desc1 = compute_lbd_descriptors(img1, lines1)
    keylines2, desc2 = compute_lbd_descriptors(img2, lines2)

    if desc1 is None or desc2 is None:
        return None

    matcher = cv2.line_descriptor.BinaryDescriptorMatcher()
    knn = matcher.knnMatch(desc1, desc2, k=2)

    raw_matches = []
    used_train = set()

    for pair in knn:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance < 0.75 * n.distance:
            if m.trainIdx in used_train:
                continue
            used_train.add(m.trainIdx)
            raw_matches.append(m)

    if len(raw_matches) < cfg.min_filtered_matches:
        return None

    A, B, matched_pairs = build_endpoint_arrays(
        raw_matches,
        keylines1,
        keylines2,
    )

    if len(A) < 8:
        return None

    return {
        "img1_raw": img1_raw,
        "img2_raw": img2_raw,
        "img1": img1,
        "img2": img2,
        "lines1": lines1,
        "lines2": lines2,
        "keylines1": keylines1,
        "keylines2": keylines2,
        "raw_matches": raw_matches,
        "filtered_matches": raw_matches,
        "A": A,
        "B": B,
        "matched_pairs": matched_pairs,
        "hist": None,
    }
