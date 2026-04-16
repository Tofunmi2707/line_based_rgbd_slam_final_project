from __future__ import annotations

"""
Run the full set of scripts used to generate dissertation report outputs.

This script provides a single orchestration entry point for the main archived
deliverables used in the final report. It executes the calibration, analysis,
figure-generation, reconstruction, and GIF-export scripts in a fixed order so
that the final outputs can be regenerated consistently from the project root.

Inspiration:
- Standard Python subprocess-based batch execution.
- The script ordering and output grouping were chosen within the present project
  to follow the structure of the dissertation results chapter.
"""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(script_path: Path) -> None:
    """
    Run a single report-generation script from the project root.

    Args:
        script_path: Path to the script that should be executed.

    Returns:
        None

    Raises:
        RuntimeError: If the script exits with a non-zero return code.
    """
    print(f"\n=== Running: {script_path.relative_to(PROJECT_ROOT)} ===")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_path}")


def main() -> None:
    """
    Execute all main report-output scripts in a fixed order.

    The execution order follows the dissertation workflow:
    calibration, front-end visualisation, odometry analysis, trajectory and loop
    closure outputs, reconstruction, and final GIF generation.

    Returns:
        None

    Raises:
        FileNotFoundError: If one or more required scripts are missing.
    """
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