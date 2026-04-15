from __future__ import annotations

from pathlib import Path

from config import (
    DATASETS,
    LoopClosureConfig,
    OdometryConfig,
    save_run_configs,
)
from src.evaluation import evaluate_trajectory
from src.loop_closure import run_loop_closure_stage
from src.odometry import run_visual_odometry
from src.reconstruction import fuse_rgbd_from_poses


def main() -> None:
    dataset_name = "fr2_large_with_loop"     # change as needed
    method_name = "v3_geom_filter"           # change as needed

    dataset_cfg = DATASETS[dataset_name]

    odom_cfg = OdometryConfig(
        method_name=method_name,
        output_dir=Path("results") / dataset_name / method_name,
    )

    loop_cfg = LoopClosureConfig()

    run_output_dir = odom_cfg.output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running dataset: {dataset_name}")
    print(f"Method: {method_name}")
    print(f"Output directory: {run_output_dir}")

    save_run_configs(
        output_dir=run_output_dir,
        dataset_cfg=dataset_cfg,
        odom_cfg=odom_cfg,
        loop_cfg=loop_cfg if dataset_name == "fr2_large_with_loop" else None,
    )

    run_out = run_visual_odometry(dataset_cfg, odom_cfg)

    if dataset_name == "fr2_large_with_loop":
        run_loop_closure_stage(
            dataset_cfg=dataset_cfg,
            odom_cfg=odom_cfg,
            loop_cfg=loop_cfg,
            poses_wc=run_out["poses_wc"],
            timestamps=run_out["timestamps"],
            image_files=run_out["image_files"],
            odometry_edges=run_out["edges"],
            output_dir=run_output_dir / loop_cfg.output_subdir,
        )

    eval_out = evaluate_trajectory(
        dataset_cfg=dataset_cfg,
        poses_wc=run_out["poses_wc"],
        timestamps=run_out["timestamps"],
        output_dir=run_output_dir,
    )

    print(f"Final RMSE: {eval_out['rmse']:.4f} m")

    try:
        fuse_rgbd_from_poses(
            dataset_cfg=dataset_cfg,
            image_files=run_out["image_files"],
            poses_wc=run_out["poses_wc"],
            output_dir=run_output_dir,
            max_frames=60,
            step=2,
            visualise=True,
        )
    except Exception as e:
        print(f"3D fusion skipped/failed: {e}")


if __name__ == "__main__":
    main()
