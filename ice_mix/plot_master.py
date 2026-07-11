import argparse
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import logging
from plot_utils import (
    calculate_angular_difference,
    calculate_vertex_distance,
    compute_statistics,
    setup_matplotlib_style,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REFERENCE_CSV = "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW.csv"


def find_result_files(base_dir):
    """Scan for the latest flat-format prediction results per project."""
    files = glob.glob(
        os.path.join(base_dir, "**", "predictions", "results.csv"), recursive=True
    )
    latest_by_project = {}

    for result_path in files:
        parsed = parse_flat_run(result_path)
        if parsed is None:
            logger.debug("Skipping legacy/non-flat prediction result: %s", result_path)
            continue

        project, job_id = parsed
        previous = latest_by_project.get(project)
        if previous is None or job_id > previous[0]:
            latest_by_project[project] = (job_id, result_path)

    return [item[1] for item in sorted(latest_by_project.values())]


def parse_flat_run(result_path):
    """Return (project, job_id) for flat run directories, else None."""
    run_dir = os.path.dirname(os.path.dirname(result_path))
    dir_name = os.path.basename(run_dir)
    match = re.match(
        r"^(?P<project>.*?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_job-(?P<job_id>\d+)$",
        dir_name,
    )
    if not match:
        return None
    return match.group("project"), int(match.group("job_id"))


def get_run_label(result_path):
    parsed = parse_flat_run(result_path)
    return parsed[0] if parsed is not None else "Unknown Run"


def plot_master_comparison(
    model_data, ref_data, metric_func, ylabel, title_prefix, output_dir, file_prefix
):
    """Plot master comparison for latest predictions plus TANGO reference."""
    modes = ["all", "tracks", "cascades"]

    for mode in modes:
        logger.info(f"Generating master plot {title_prefix} ({mode})...")

        plt.figure(figsize=(16, 8))

        # Plot models
        for label, df in model_data:
            df_mode = filter_data_by_mode(df, mode)
            if df_mode.empty:
                continue

            values = metric_func(df_mode)
            df_mode["metric"] = values
            stats = compute_statistics(df_mode, "metric")

            if stats:
                plt.plot(
                    stats["centers"],
                    stats["median"],
                    label=label,
                    marker="o",
                    alpha=0.7,
                )
                plt.fill_between(
                    stats["centers"], stats["lower"], stats["upper"], alpha=0.1
                )

        if ref_data is not None:
            ref_mode = filter_data_by_mode(ref_data, mode)
            if not ref_mode.empty:
                ref_values = metric_func(ref_mode)
                ref_mode["metric"] = ref_values
                ref_stats = compute_statistics(ref_mode, "metric")

                if ref_stats:
                    plt.plot(
                        ref_stats["centers"],
                        ref_stats["median"],
                        label="TANGO (Ref)",
                        color="black",
                        linewidth=3,
                        linestyle="--",
                    )

        plt.xlabel("log10(Energy [GeV])")
        plt.ylabel(ylabel)
        plt.title(f"Master Plot: {title_prefix} - {mode.capitalize()}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, f"master_{file_prefix}_{mode}.png"), dpi=300
        )
        plt.close()


def filter_data_by_mode(df, mode):
    """Helper to filter dataframe by mode."""
    if mode == "all":
        return df.copy()
    elif mode == "tracks":
        return df[(abs(df["pid"]) == 14) & (df["interaction_type"] == 1)].copy()
    elif mode == "cascades":
        return df[~((abs(df["pid"]) == 14) & (df["interaction_type"] == 1))].copy()
    return df.copy()


def remove_stale_old_plots(output_dir):
    for path in glob.glob(os.path.join(output_dir, "*_no_old_*.png")):
        try:
            os.remove(path)
            logger.info("Removed stale OLD-filtered plot: %s", path)
        except OSError as e:
            logger.warning("Could not remove stale plot %s: %s", path, e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir", required=True, help="Base directory containing run outputs"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save master plots"
    )
    args = parser.parse_args()

    setup_matplotlib_style()
    os.makedirs(args.output_dir, exist_ok=True)
    remove_stale_old_plots(args.output_dir)

    # 1. Load TANGO Reference
    logger.info(f"Loading TANGO reference from {REFERENCE_CSV}")
    ref_df = None
    if os.path.exists(REFERENCE_CSV):
        ref_df = pd.read_csv(REFERENCE_CSV)
    else:
        logger.warning("TANGO reference file not found.")

    # 2. Find and Load Model Results
    result_files = find_result_files(args.base_dir)
    logger.info(f"Found {len(result_files)} latest result files.")

    model_data = []
    for f in result_files:
        label = get_run_label(f)
        if label == "Unknown Run":
            continue
        try:
            df = pd.read_csv(f)
            model_data.append((label, df))
        except Exception as e:
            logger.error(f"Error loading {f}: {e}")

    if not model_data:
        logger.warning("No model data found. Exiting.")
        return

    # 3. Angular Resolution Master Plots
    plot_master_comparison(
        model_data,
        ref_df,
        lambda d: calculate_angular_difference(
            d["azimuth"], d["zenith"], d["dir_x_pred"], d["dir_y_pred"], d["dir_z_pred"]
        ),
        "Angular Error [deg]",
        "Angular Resolution",
        args.output_dir,
        "angular_res",
    )

    # 4. Vertex Resolution Master Plots
    plot_master_comparison(
        model_data,
        ref_df,
        lambda d: calculate_vertex_distance(d),
        "Vertex Error [m]",
        "Vertex Resolution",
        args.output_dir,
        "vertex_res",
    )


if __name__ == "__main__":
    main()
