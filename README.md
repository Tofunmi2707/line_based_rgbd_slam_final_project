
---

## Main `README.md`

```markdown
# Line-based RGB-D SLAM for low-texture indoor environments

This repository contains the code, experiment scripts, and selected output structure for a third-year project on a **line-based RGB-D visual SLAM pipeline** for low-texture indoor environments using the **TUM RGB-D benchmark**.

The project investigates how line detection, preprocessing, descriptor-based matching, geometric filtering, calibrated two-view pose estimation, loop-closure experimentation, and RGB-D fusion affect odometry, trajectory quality, and 3D reconstruction.

## Project scope

The implemented system is a staged RGB-D SLAM-style pipeline focused on:

- line detection with LSD
- optional CLAHE preprocessing
- three front-end odometry variants:
  - **V1:** centroid-based matching with centroid pose input
  - **V2:** LBD-based matching with endpoint correspondences
  - **V3:** LBD-based matching with endpoint correspondences plus geometric filtering
- Essential-matrix pose estimation with RANSAC
- depth-assisted metric translation estimation
- trajectory alignment and RMSE evaluation
- RGB-D point-cloud fusion
- pose-graph loop-closure experimentation on `fr2_large_with_loop`

## Benchmark datasets used

The reported experiments use these TUM RGB-D sequences:

- `fr1_desk`
- `fr1_room`
- `fr1_xyz`
- `fr2_large_with_loop`

See `data/README.md` for the expected folder structure and dataset notes.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── main.py
├── config.py
├── data/
│   └── README.md
├── src/
│   ├── tum_io.py
│   ├── line_frontend_v1_centroid.py
│   ├── line_frontend_v2_lbd_endpoints.py
│   ├── line_frontend_v3_geom_filter.py
│   ├── pose_estimation.py
│   ├── odometry.py
│   ├── evaluation.py
│   ├── reconstruction.py
│   ├── pose_graph_2d.py
│   └── loop_closure.py
├── scripts/
│   ├── run_all_report_outputs.py
│   ├── calibration/
│   │   ├── run_camera_calibration.py
│   │   └── images/
│   ├── figures/
│   │   ├── make_clahe_comparison.py
│   │   ├── make_cloud_gif.py
│   │   ├── make_frontend_visuals.py
│   │   ├── make_reconstruction_figures.py
│   │   ├── make_single_frame_cloud.py
│   │   ├── make_trajectory_plots.py
│   │   └── run_fusion_sensitivity.py
│   └── analysis/
│       ├── summarise_odometry_debug.py
│       └── summarise_loop_closure.py
├── results/
└── docs/