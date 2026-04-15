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


def detect_lines_lsd(img: np.ndarray, max_lines: int) -> np.ndarray:
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]
    if lines is None:
        return np.empty((0, 1, 4), dtype=np.float32)

    xy = lines.reshape(-1, 4)
    lengths = np.hypot(xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1])
    idx = np.argsort(lengths)[::-1][:max_lines]
    return lines[idx].reshape(-1, 1, 4)


def lines_to_keylines(lines: np.ndarray):
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


def compute_lbd_descriptors(img: np.ndarray, lines: np.ndarray):
    keylines = lines_to_keylines(lines)

    bd = cv2.line_descriptor.BinaryDescriptor.createBinaryDescriptor()
    keylines, desc = bd.compute(img, keylines)

    if desc is None or len(keylines) == 0:
        return [], None

    return keylines, desc


def endpoint_pair_from_match(kl1, kl2):
    A = np.array([
        [kl1.startPointX, kl1.startPointY],
        [kl1.endPointX, kl1.endPointY],
    ], dtype=np.float32)

    B = np.array([
        [kl2.startPointX, kl2.startPointY],
        [kl2.endPointX, kl2.endPointY],
    ], dtype=np.float32)

    d_same = np.linalg.norm(A[0] - B[0]) + np.linalg.norm(A[1] - B[1])
    d_swap = np.linalg.norm(A[0] - B[1]) + np.linalg.norm(A[1] - B[0])

    if d_swap < d_same:
        B = B[::-1]

    return A, B


def build_endpoint_arrays(matches, keylines1, keylines2):
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


def draw_line_matches(gray1: np.ndarray,
                      gray2: np.ndarray,
                      keylines1,
                      keylines2,
                      matches,
                      max_draw: int | None = 12,
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

    draw_matches = sorted(matches, key=lambda m: m.distance)
    draw_matches = (
        draw_matches if max_draw is None else draw_matches[:max_draw]
    )
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

        m1 = (int(round(kl1.pt[0])), int(round(kl1.pt[1])))
        m2 = (int(round(kl2.pt[0] + w1)), int(round(kl2.pt[1])))

        cv2.line(canvas, m1, m2, colour, line_thickness, cv2.LINE_AA)
        cv2.circle(canvas, m1, 5, colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, m2, 5, colour, -1, cv2.LINE_AA)

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

    raw_match_vis = draw_line_matches(
        result["img1"],
        result["img2"],
        result["keylines1"],
        result["keylines2"],
        result["raw_matches"],
        max_draw=20
    )
    cv2.imwrite(str(output_dir / "raw_matches.png"), raw_match_vis)

    return result


def process_frame_pair_frontend(img1_path: str,
                                img2_path: str,
                                cfg) -> dict | None:
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
        keylines2
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
