from __future__ import annotations

# Inspiration: standard Python subprocess workflow for reproducible batch runs;
# project-specific orchestration and output ordering implemented for this project.

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(script_path: Path) -> None:
    # Inspiration: standard subprocess.run pattern; wrapped here so all report
    # generation scripts run from the project root in a fixed order.
    print(f"\n=== Running: {script_path.relative_to(PROJECT_ROOT)} ===")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_path}")


def main() -> None:
    # Inspiration: batch generation pipeline; order chosen to
    # match the report narrative from calibration to frontend to odometry to
    # trajectory to loop closure to reconstruction and GIF outputs.
    scripts = [
        PROJECT_ROOT / "scripts" / "calibration" / "run_camera_calibration.py",
        PROJECT_ROOT / "scripts" / "figures" / "make_frontend_visuals.py",
        PROJECT_ROOT / "scripts" / "figures" / "make_clahe_comparison.py",
        PROJECT_ROOT / "scripts" / "analysis" / "summarise_odometry_debug.py",
        PROJECT_ROOT / "scripts" / "figures" / "make_trajectory_plots.py",
        PROJECT_ROOT / "scripts" / "analysis" / "summarise_loop_closure.py",
        PROJECT_ROOT / "scripts" / "figures" / "make_single_frame_cloud.py",
        PROJECT_ROOT / "scripts" / "figures" / "run_fusion_sensitivity.py",
        PROJECT_ROOT / "scripts" / "figures" / "make_reconstruction_figures.py",
        PROJECT_ROOT / "scripts" / "figures" / "make_cloud_gif.py",
    ]

    missing = [p for p in scripts if not p.exists()]
    if missing:
        print("Missing script files:")
        for p in missing:
            print(f"  - {p}")
        raise FileNotFoundError("Create the missing script files first.")

    for script in scripts:
        run_script(script)

    print("\nAll report outputs generated successfully.")


if __name__ == "__main__":
    main()