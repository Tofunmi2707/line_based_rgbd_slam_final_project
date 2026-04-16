# Data folder

This project uses sequences from the **TUM RGB-D benchmark** for all reported odometry, trajectory, loop-closure, and reconstruction experiments.

## Datasets used in the dissertation

- `fr1_desk`
- `fr1_room`
- `fr1_xyz`
- `fr2_large_with_loop`

## Important note

The benchmark datasets are **not redistributed** in this repository.  
They should be downloaded from the official TUM RGB-D dataset source and placed into this `data/` folder using the directory structure expected by the code.

## Expected directory structure

```text
data/
  fr1_desk/
    rgb/
    depth/
    groundtruth.txt
    rgb_depth_assoc.txt   # if generated/used
  fr1_room/
    rgb/
    depth/
    groundtruth.txt
    rgb_depth_assoc.txt   # if generated/used
  fr1_xyz/
    rgb/
    depth/
    groundtruth.txt
    rgb_depth_assoc.txt   # if generated/used
  fr2_large_with_loop/
    rgb/
    depth/
    groundtruth.txt
    rgb_depth_assoc.txt   # if generated/used