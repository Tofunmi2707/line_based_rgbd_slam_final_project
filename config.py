from __future__ import annotations

"""
Configuration definitions for the line-based RGB-D SLAM project.

This module defines:
- dataset configuration records,
- odometry/front-end configuration records,
- loop-closure configuration records,
- dataset-specific TUM RGB-D benchmark intrinsics,
- helper utilities for saving run configurations alongside results.

The configuration objects are used to keep reported experiments reproducible
and to separate odometry settings from backend loop-closure settings.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np


@dataclass
class DatasetConfig:
    """
    Store dataset paths and camera parameters for a benchmark sequence.

    Attributes:
        name: Dataset name used throughout the project.
        dataset_dir: Root directory of the dataset.
        rgb_dir: Directory containing RGB frames.
        depth_dir: Directory containing depth frames.
        groundtruth_path: Path to benchmark ground-truth trajectory.
        assoc_path: Optional RGB-depth association file.
        intrinsics: Camera intrinsic matrix used for projection
        /back-projection.
        depth_scale: Depth scaling factor used by the dataset.
        max_rgb_depth_dt: Maximum timestamp difference allowed for
        RGB-depth pairing.
        freiburg_group: Freiburg sensor group label.
        depth_correction_factor: Dataset-specific depth correction factor.
    """
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
    """
    Store front-end and odometry settings for the reported experiments.

    The same core acceptance thresholds are intended to be reused across the
    odometry comparison runs so that differences between V1, V2, and V3 arise
    from correspondence representation and filtering strategy rather than from
    changing acceptance rules.
    """

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
        """
        Convert the odometry configuration to a JSON-safe dictionary.

        Returns:
            Dictionary representation of the configuration with paths converted
            to strings where needed.
        """
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass
class LoopClosureConfig:
    """
    Store backend loop-closure and pose-graph optimisation settings.

    These settings are kept separate from the odometry configuration so
    that the loop-closure experiment can be reported as a distinct backend
    stage.
    """
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
        """
        Convert the loop-closure configuration to a JSON-safe dictionary.

        Returns:
            Dictionary representation of the loop-closure configuration.
        """
        return asdict(self)


def _freiburg1_intrinsics() -> np.ndarray:
    """
    Return the TUM RGB-D Freiburg 1 colour-camera intrinsics.

    Returns:
        3 x 3 camera intrinsic matrix for Freiburg 1 sequences.
    """
    return np.array(
        [
            [517.3, 0.0, 318.6],
            [0.0, 516.5, 255.3],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _freiburg2_intrinsics() -> np.ndarray:
    """
    Return the TUM RGB-D Freiburg 2 colour-camera intrinsics.

    Returns:
        3 x 3 camera intrinsic matrix for Freiburg 2 sequences.
    """
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
    """
    Build a dataset configuration for a TUM RGB-D sequence.

    Args:
        name: Dataset name.
        dataset_dir: Root folder of the dataset.
        intrinsics: Intrinsic camera matrix for the sequence.
        freiburg_group: Freiburg sensor group label.
        depth_correction_factor: Dataset-specific depth correction factor.

    Returns:
        DatasetConfig populated with the expected TUM folder structure.
    """
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
    """
    Save the configurations used for a run into the output directory.

    This function writes separate JSON files for:
    - dataset configuration,
    - odometry/front-end configuration,
    - loop-closure configuration, when used.

    Args:
        output_dir: Directory in which the configuration files should be saved.
        dataset_cfg: Dataset configuration used for the run.
        odom_cfg: Odometry configuration used for the run.
        loop_cfg: Optional loop-closure configuration used for backend
        evaluation.

    Returns:
        None
    """
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

    with open(
        output_dir / "dataset_config_used.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(dataset_dict, f, indent=2)

    with open(
        output_dir / "odometry_config_used.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(odom_cfg.to_serialisable_dict(), f, indent=2)

    if loop_cfg is not None:
        with open(
            output_dir / "loop_closure_config_used.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(loop_cfg.to_serialisable_dict(), f, indent=2)
