import argparse
import os
import glob
import subprocess
import logging
import sys
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("generate_plots")


def find_result_files(base_dir):
    """Scan for the latest flat-format prediction results per project."""
    search_pattern = os.path.join(base_dir, "**", "predictions", "results.csv")
    files = glob.glob(search_pattern, recursive=True)
    latest_by_project = {}

    for result_csv in files:
        run_dir = os.path.dirname(os.path.dirname(result_csv))
        run_name = os.path.basename(run_dir)
        match = re.match(
            r"^(?P<project>.*?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_job-(?P<job_id>\d+)$",
            run_name,
        )
        if not match:
            logger.debug("Skipping legacy/non-flat prediction result: %s", result_csv)
            continue

        project = match.group("project")
        job_id = int(match.group("job_id"))
        previous = latest_by_project.get(project)
        if previous is None or job_id > previous[0]:
            latest_by_project[project] = (job_id, result_csv)

    return [item[1] for item in sorted(latest_by_project.values())]


def run_plot_command(script_name, args):
    """Execute a python plotting script with arguments."""
    # Assume script is in the same directory as this script or in python path
    # We'll try to find it relative to this script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)

    if not os.path.exists(script_path):
        # Fallback: maybe we are running from root and script is in ice_mix/
        script_path = os.path.join("ice_mix", script_name)

    if not os.path.exists(script_path):
        logger.error(f"Could not find script: {script_name}")
        return False

    cmd = [sys.executable, script_path] + args

    try:
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate all plots for IceMix predictions."
    )
    parser.add_argument(
        "--base-dir",
        default="ice_mix/outputs",
        help="Base directory to scan for results (default: ice_mix/outputs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of plots even if they exist",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    if not os.path.exists(base_dir):
        logger.error(f"Base directory does not exist: {base_dir}")
        return

    # 1. Scan for results
    logger.info(f"Scanning {base_dir} for results.csv files...")
    result_files = find_result_files(base_dir)

    if not result_files:
        logger.warning("No results.csv files found! Have you run predictions yet?")
        logger.warning("Try running: sbatch ice_mix/run_predict.sbatch")
        return

    logger.info(f"Found {len(result_files)} result files.")

    # 2. Process each run (Per-Run Plots)
    for result_csv in result_files:
        # result_csv is like .../predictions/results.csv
        # run_dir is likely .../ (parent of predictions) or just the directory containing results
        # Based on predict.py: output_dir is where results.csv is. run_dir is passed to predict.py.
        # But predict.py saves results in {run_dir}/predictions/results.csv (or similar)

        # Let's derive run_dir.
        # If path is .../run_name/predictions/results.csv, run_dir is .../run_name
        predictions_dir = os.path.dirname(result_csv)
        run_dir = os.path.dirname(predictions_dir)

        # Output dir for plots
        plots_dir = os.path.join(run_dir, "plots")

        logger.info(f"Processing run: {run_dir}")

        # Call plot_run.py
        success = run_plot_command(
            "plot_run.py",
            [
                "--results-csv",
                result_csv,
                "--run-dir",
                run_dir,
                "--output-dir",
                plots_dir,
            ],
        )

        if success:
            logger.info(f"Plots generated in {plots_dir}")
        else:
            logger.warning(f"Failed to generate plots for {run_dir}")

    # 3. Process Aggregate (Master Plots)
    logger.info("Generating Master Plots...")
    master_plots_dir = os.path.join(base_dir, "master_plots")

    success = run_plot_command(
        "plot_master.py", ["--base-dir", base_dir, "--output-dir", master_plots_dir]
    )

    if success:
        logger.info(f"Master plots generated in {master_plots_dir}")
    else:
        logger.error("Failed to generate master plots")

    # 4. Process Aggregate (Reference Plots)
    logger.info("Generating Reference Plots...")
    reference_plots_dir = os.path.join(base_dir, "reference_plot")

    success = run_plot_command(
        "plot_reference.py",
        ["--base-dir", base_dir, "--output-dir", reference_plots_dir],
    )

    if success:
        logger.info(f"Reference plots generated in {reference_plots_dir}")
    else:
        logger.error("Failed to generate reference plots")

    logger.info("All plots generated!")


if __name__ == "__main__":
    main()
