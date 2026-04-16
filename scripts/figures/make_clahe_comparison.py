from __future__ import annotations

# Inspiration: OpenCV CLAHE preprocessing and LSD line detection APIs;
# project-specific frame sampling, line-count comparison, and dissertation
# figure/table export are implemented here.

from pathlib import Path
import sys
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS  # noqa: E402
from src.tum_io import load_rgb_sequence  # noqa: E402


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    # Inspiration: OpenCV CLAHE interface; parameters exposed here so the
    # preprocessing settings used for the report are explicit.
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(img)


def detect_lines_lsd(img: np.ndarray) -> np.ndarray:
    # Inspiration: OpenCV LSD detector interface; wrapped so the output is
    # always safe to count, even when no lines are returned.
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]
    if lines is None:
        return np.empty((0, 1, 4), dtype=np.float32)
    return lines


def count_lines(img: np.ndarray) -> int:
    return len(detect_lines_lsd(img))


def save_side_by_side(
    original: np.ndarray,
    enhanced: np.ndarray,
    out_path: Path,
    title_left: str = "Original grayscale",
    title_right: str = "CLAHE",
) -> None:
    # Inspiration: matplotlib side-by-side comparison layout for report figures.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title(title_left)
    axes[0].axis("off")

    axes[1].imshow(enhanced, cmap="gray")
    axes[1].set_title(title_right)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_count_plot(
    frame_indices: list[int],
    counts_raw: list[int],
    counts_clahe: list[int],
    out_path: Path,
    dataset_name: str,
) -> None:
    # Inspiration: grouped bar chart for direct no-CLAHE vs CLAHE comparison.
    x = np.arange(len(frame_indices))
    width = 0.38

    plt.figure(figsize=(11, 5))
    plt.bar(
        x - width / 2,
        counts_raw,
        width=width,
        label="Without CLAHE",
        edgecolor="black",
        linewidth=0.8,
    )
    plt.bar(
        x + width / 2,
        counts_clahe,
        width=width,
        label="With CLAHE",
        edgecolor="black",
        linewidth=0.8,
    )

    plt.xticks(x, [str(i) for i in frame_indices])
    plt.xlabel("Frame index")
    plt.ylabel("Number of LSD line segments")
    plt.title(f"LSD line counts on {dataset_name} with and without CLAHE")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    # Inspiration: dissertation figure generation for the preprocessing section.
    # fr1_room is used because it provides a clearer hard-case frontend example.
    dataset_name = "fr1_room"
    frame_indices = [20, 40, 60, 80, 100, 120]

    dataset_cfg = DATASETS[dataset_name]
    rgb_files = load_rgb_sequence(dataset_cfg.rgb_dir)

    out_dir = PROJECT_ROOT / "results" / dataset_name / "clahe_line_count_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[list[int]] = []
    counts_raw: list[int] = []
    counts_clahe: list[int] = []

    for idx in frame_indices:
        if idx < 0 or idx >= len(rgb_files):
            print(f"Skipping frame {idx}: out of range")
            continue

        img_bgr = cv2.imread(str(rgb_files[idx]))
        if img_bgr is None:
            print(f"Skipping frame {idx}: could not read image")
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe_img = apply_clahe(gray)

        raw_count = count_lines(gray)
        clahe_count = count_lines(clahe_img)
        diff = clahe_count - raw_count

        counts_raw.append(raw_count)
        counts_clahe.append(clahe_count)
        rows.append([idx, raw_count, clahe_count, diff])

        save_side_by_side(
            gray,
            clahe_img,
            out_dir / f"frame_{idx}_clahe_comparison.png",
        )

        print(
            f"frame {idx}: without CLAHE = {raw_count}, "
            f"with CLAHE = {clahe_count}, diff = {diff:+d}"
        )

    csv_path = out_dir / f"clahe_line_counts_{dataset_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_idx",
                "lines_without_clahe",
                "lines_with_clahe",
                "difference",
            ]
        )
        writer.writerows(rows)

    if rows:
        valid_indices = [r[0] for r in rows]

        save_count_plot(
            valid_indices,
            counts_raw,
            counts_clahe,
            out_dir / f"clahe_line_count_plot_{dataset_name}.png",
            dataset_name,
        )

        mean_raw = float(np.mean(counts_raw))
        mean_clahe = float(np.mean(counts_clahe))
        mean_diff = mean_clahe - mean_raw

        summary_path = out_dir / "clahe_line_count_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Frames analysed: {valid_indices}\n")
            f.write(f"Mean lines without CLAHE: {mean_raw:.2f}\n")
            f.write(f"Mean lines with CLAHE: {mean_clahe:.2f}\n")
            f.write(f"Mean difference: {mean_diff:.2f}\n")

        print("\nSummary")
        print("-------")
        print(f"Mean lines without CLAHE: {mean_raw:.2f}")
        print(f"Mean lines with CLAHE   : {mean_clahe:.2f}")
        print(f"Mean difference         : {mean_diff:.2f}")

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
