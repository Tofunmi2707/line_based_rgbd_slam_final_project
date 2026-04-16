from __future__ import annotations

# Inspiration: project frontend visualisation workflow built around the
# implemented V3 geometric-filter pipeline; raw/rejected match rendering and
# report-specific file organisation are handled here for dissertation figures.

from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS, OdometryConfig  # noqa: E402
from src.tum_io import load_rgb_sequence  # noqa: E402
from src.line_frontend_v3_geom_filter import save_frontend_visuals  # noqa: E402
from src.line_frontend_v3_geom_filter import subtract_matches  # noqa: E402
from src.line_frontend_v3_geom_filter import draw_line_matches  # noqa: E402


def save_rejected_match_visual(
    result: dict,
    out_dir: Path,
    filename: str = "rejected_matches.png",
) -> Path:
    # Inspiration: project-specific separation of raw/filtered/rejected match
    # sets so the filtering effect can be shown honestly in the report.
    rejected_matches = subtract_matches(
        result["raw_matches"],
        result["filtered_matches"],
    )

    rejected_match_vis = draw_line_matches(
        result["img1"],
        result["img2"],
        result["keylines1"],
        result["keylines2"],
        rejected_matches,
        max_draw=None,
    )

    out_path = out_dir / filename
    cv2.imwrite(str(out_path), rejected_match_vis)
    return out_path


def main() -> None:
    # Inspiration: dissertation figure generation for the frontend evidence
    # subsection; fr1_room is used because it gives the clearest detector and
    # filtering visuals in the final report.
    dataset_name = "fr1_room"
    frame_idx = 100

    dataset_cfg = DATASETS[dataset_name]
    cfg = OdometryConfig()

    rgb_files = load_rgb_sequence(dataset_cfg.rgb_dir)

    if frame_idx + 1 >= len(rgb_files):
        raise IndexError(
            f"frame_idx {frame_idx} is too large for dataset {dataset_name}"
        )

    out_dir = PROJECT_ROOT / "results" / "frontend_visuals" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = save_frontend_visuals(
        str(rgb_files[frame_idx]),
        str(rgb_files[frame_idx + 1]),
        cfg,
        out_dir,
    )

    if result is None:
        print("Frontend visualisation failed for this frame pair.")
        return

    rejected_out = save_rejected_match_visual(result, out_dir)

    print(f"Saved frontend visuals to: {out_dir}")
    print(f"Raw matches     : {len(result['raw_matches'])}")
    print(f"Filtered matches: {len(result['filtered_matches'])}")
    print(
        f"Rejected matches: "
        f"{len(result['raw_matches']) - len(result['filtered_matches'])}"
    )
    print(f"Rejected match visual: {rejected_out}")


if __name__ == "__main__":
    main()
