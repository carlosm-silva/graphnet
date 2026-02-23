import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import logging
from plot_utils import (
    load_and_filter_data,
    calculate_angular_difference,
    calculate_vertex_distance,
    compute_statistics,
    setup_matplotlib_style,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REFERENCE_CSV = "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW.csv"


def find_result_files(base_dir):
    """Scan directory for results.csv files."""
    # Pattern: base_dir/YYYY-MM-DD/HH-MM-SS/predictions/results.csv
    files = glob.glob(
        os.path.join(base_dir, "**", "predictions", "results.csv"), recursive=True
    )
    return files


def get_run_label(result_path):
    """Generate a label for the run based on config.yaml."""
    try:
        import yaml

        run_dir = os.path.dirname(os.path.dirname(result_path))
        config_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config and "project_name" in config:
                    return config["project_name"]
    except Exception as e:
        logger.warning(f"Could not load run label from config: {e}")

    # Fallback to date/time format
    parts = result_path.split(os.sep)
    if len(parts) >= 4:
        return f"{parts[-4]}/{parts[-3]}"
    return "Unknown Run"


def plot_master_comparison(
    model_data, ref_data, metric_func, ylabel, title_prefix, output_dir, file_prefix
):
    """Plot master comparison for all models + reference."""
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

        # Plot Reference
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
        return df
    elif mode == "tracks":
        return df[(abs(df["pid"]) == 14) & (df["interaction_type"] == 1)]
    elif mode == "cascades":
        return df[~((abs(df["pid"]) == 14) & (df["interaction_type"] == 1))]
    return df


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

    # 1. Load Reference
    logger.info(f"Loading reference from {REFERENCE_CSV}")
    ref_df = None
    if os.path.exists(REFERENCE_CSV):
        ref_df = pd.read_csv(REFERENCE_CSV)
    else:
        logger.warning("Reference file not found!")

    # 2. Find and Load Model Results
    result_files = find_result_files(args.base_dir)
    logger.info(f"Found {len(result_files)} result files.")

    model_data = []
    for f in result_files:
        label = get_run_label(f)
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
