from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np


@dataclass
class DatasetConfig:
    name: str
    dataset_dir: Path
    rgb_dir: Path
    depth_dir: Path
    groundtruth_path: Path
    assoc_path: Path | None
    intrinsics: np.ndarray
    depth_scale: float = 5000.0
    max_rgb_depth_dt: float = 0.03
    freiburg_group: str = "unknown"
    depth_correction_factor: float = 1.0


@dataclass
class OdometryConfig:
    # Front-end / odometry settings
    max_lines: int = 1500

    # V1 centroid matcher settings
    grid_cell_size: int = 40
    max_angle_diff_deg: float = 15.0
    max_centroid_dist_px: float = 80.0
    min_length_ratio: float = 0.5
    max_length_ratio: float = 2.0
    match_score_threshold: float = 40.0

    # V1 histogram filter
    hist_bins: int = 30
    length_band_width: float = 10.0
    angle_band_width_deg: float = 10.0

    # Shared pose acceptance thresholds
    min_filtered_matches: int = 12
    min_essential_inliers: int = 10
    min_metric_points: int = 6
    max_step_metres: float = 0.15

    # Reporting / output
    output_dir: Path = Path("results")
    method_name: str = "v2_lbd_endpoints"

    def to_serialisable_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass
class LoopClosureConfig:
    min_frame_gap: int = 80
    pose_radius: float = 0.40
    max_candidates_per_frame: int = 1
    max_loop_step_metres: float = 1.0

    # Pose-graph optimisation settings
    iters: int = 10
    w_odo: float = 1.0
    w_loop: float = 3.0

    # Output
    output_subdir: str = "loop_closure"

    def to_serialisable_dict(self) -> dict[str, Any]:
        return asdict(self)


def _freiburg1_intrinsics() -> np.ndarray:
    return np.array(
        [
            [517.3, 0.0, 318.6],
            [0.0, 516.5, 255.3],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _freiburg2_intrinsics() -> np.ndarray:
    return np.array(
        [
            [520.9, 0.0, 325.1],
            [0.0, 521.0, 249.7],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def make_tum_dataset(
    name: str,
    dataset_dir: str,
    intrinsics: np.ndarray,
    freiburg_group: str,
    depth_correction_factor: float,
) -> DatasetConfig:
    root = Path(dataset_dir)
    assoc_file = root / "rgb_depth_assoc.txt"
    assoc_path = assoc_file if assoc_file.exists() else None

    return DatasetConfig(
        name=name,
        dataset_dir=root,
        rgb_dir=root / "rgb",
        depth_dir=root / "depth",
        groundtruth_path=root / "groundtruth.txt",
        assoc_path=assoc_path,
        intrinsics=intrinsics,
        freiburg_group=freiburg_group,
        depth_correction_factor=depth_correction_factor,
    )


DATASETS: dict[str, DatasetConfig] = {
    "fr1_xyz": make_tum_dataset(
        name="fr1_xyz",
        dataset_dir="data/fr1_xyz",
        intrinsics=_freiburg1_intrinsics(),
        freiburg_group="Freiburg 1",
        depth_correction_factor=1.035,
    ),
    "fr1_desk": make_tum_dataset(
        name="fr1_desk",
        dataset_dir="data/fr1_desk",
        intrinsics=_freiburg1_intrinsics(),
        freiburg_group="Freiburg 1",
        depth_correction_factor=1.035,
    ),
    "fr1_room": make_tum_dataset(
        name="fr1_room",
        dataset_dir="data/fr1_room",
        intrinsics=_freiburg1_intrinsics(),
        freiburg_group="Freiburg 1",
        depth_correction_factor=1.035,
    ),
    "fr2_large_with_loop": make_tum_dataset(
        name="fr2_large_with_loop",
        dataset_dir="data/fr2_large_with_loop",
        intrinsics=_freiburg2_intrinsics(),
        freiburg_group="Freiburg 2",
        depth_correction_factor=1.031,
    ),
}


def save_run_configs(
    output_dir: Path,
    dataset_cfg: DatasetConfig,
    odom_cfg: OdometryConfig,
    loop_cfg: LoopClosureConfig | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = {
        "name": dataset_cfg.name,
        "dataset_dir": str(dataset_cfg.dataset_dir),
        "rgb_dir": str(dataset_cfg.rgb_dir),
        "depth_dir": str(dataset_cfg.depth_dir),
        "groundtruth_path": str(dataset_cfg.groundtruth_path),
        "assoc_path": str(dataset_cfg.assoc_path)
        if dataset_cfg.assoc_path else None,
        "intrinsics": dataset_cfg.intrinsics.tolist(),
        "depth_scale": dataset_cfg.depth_scale,
        "max_rgb_depth_dt": dataset_cfg.max_rgb_depth_dt,
        "freiburg_group": dataset_cfg.freiburg_group,
        "depth_correction_factor": dataset_cfg.depth_correction_factor,
    }

    with open(output_dir / "dataset_config_used.json",
              "w",
              encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2)

    with open(output_dir / "odometry_config_used.json",
              "w",
              encoding="utf-8") as f:
        json.dump(odom_cfg.to_serialisable_dict(), f, indent=2)

    if loop_cfg is not None:
        with open(output_dir / "loop_closure_config_used.json",
                  "w",
                  encoding="utf-8") as f:
            json.dump(loop_cfg.to_serialisable_dict(), f, indent=2)
