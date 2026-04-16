from __future__ import annotations

# Inspiration: project RGB-D fusion workflow built on the implemented
# reconstruction module; frame-window sweep and report-specific output
# organisation are handled here for the fusion-sensitivity experiment.

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS  # noqa: E402
from src.reconstruction import fuse_rgbd_from_poses  # noqa: E402


def main() -> None:
    # Inspiration: dissertation fusion-sensitivity experiment; fr1_desk is used
    # because it provides the clearest qualitative comparison across frame windows.
    dataset_name = "fr1_desk"
    method_name = "v3_geom_filter"
    frame_settings = [10, 30, 60, 90, 120, 150, 180]

    dataset_cfg = DATASETS[dataset_name]
    run_dir = PROJECT_ROOT / "results" / dataset_name / method_name

    poses_path = run_dir / "poses_3d.npz"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing poses file: {poses_path}")

    data = np.load(poses_path, allow_pickle=True)
    poses_wc = data["poses_wc"]
    image_files = list(data["image_files"])

    summary_lines = []
    summary_lines.append(f"Dataset: {dataset_name}")
    summary_lines.append(f"Method: {method_name}")
    summary_lines.append(f"Frame settings tested: {frame_settings}")
    summary_lines.append("")

    for max_frames in frame_settings:
        out_dir = run_dir / f"fusion_{max_frames}_frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning fusion with max_frames={max_frames}")

        try:
            fuse_rgbd_from_poses(
                dataset_cfg=dataset_cfg,
                image_files=image_files,
                poses_wc=poses_wc,
                output_dir=out_dir,
                max_frames=max_frames,
                step=2,
                visualise=False,
            )
            summary_lines.append(
                f"{max_frames} frames: success -> outputs saved to {out_dir}"
            )
        except Exception as e:
            print(f"Fusion failed for {max_frames} frames: {e}")
            summary_lines.append(f"{max_frames} frames: failed -> {e}")

    summary_path = run_dir / "fusion_sensitivity_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSaved fusion sensitivity summary to: {summary_path}")


if __name__ == "__main__":
    main()
