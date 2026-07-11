import argparse
import os
import glob
import re
import pandas as pd
import numpy as np
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


def plot_loss(run_dir, output_dir):
    """Plot training and validation loss."""
    # Look for CSV logs first
    # Standard location: run_dir/logs/training_logs/version_0/metrics.csv
    # Or run_dir/../logs/training_logs... depending on how Hydra set it up.
    # We'll search for metrics.csv

    metrics_path = None
    for root, dirs, files in os.walk(run_dir):
        if "metrics.csv" in files:
            metrics_path = os.path.join(root, "metrics.csv")
            break

    if not metrics_path:
        logger.warning(f"No metrics.csv found in {run_dir}. Skipping loss plot.")
        return

    try:
        df = pd.read_csv(metrics_path)

        plt.figure(figsize=(10, 6))

        train_col = None
        if "train_loss_epoch" in df.columns:
            train_col = "train_loss_epoch"
        elif "train_loss" in df.columns:
            train_col = "train_loss"

        if train_col:
            # Drop NaNs for plotting lines properly
            train_df = df.dropna(subset=[train_col])
            plt.plot(
                train_df["epoch"],
                train_df[train_col],
                label="Train Loss",
                marker="o",
            )

        if "val_loss" in df.columns:
            val_df = df.dropna(subset=["val_loss"])
            plt.plot(val_df["epoch"], val_df["val_loss"], label="Val Loss", marker="s")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        plt.savefig(os.path.join(output_dir, "loss_plot.png"))
        plt.close()
        logger.info("Loss plot created.")

    except Exception as e:
        logger.error(f"Failed to plot loss: {e}")


def plot_resolution(
    df,
    ref_df,
    metric_func,
    ylabel,
    title_prefix,
    output_dir,
    file_prefix,
    run_label="IceMix (This Run)",
    ref_label="IceMix nu_tau Baseline",
):
    """Generic resolution plotting function."""
    modes = ["all", "tracks", "cascades"]

    for mode in modes:
        logger.info(f"Plotting {title_prefix} for {mode}...")

        # Filter data
        df_mode = filter_data_by_mode(df, mode)
        ref_df_mode = filter_data_by_mode(ref_df, mode)

        if df_mode.empty:
            logger.warning(f"No data for {mode}, skipping.")
            continue

        # Calculate metric
        values = metric_func(df_mode)
        ref_values = metric_func(ref_df_mode)

        # Add to dataframe for statistics
        df_mode["metric"] = values
        ref_df_mode["metric"] = ref_values

        # Compute stats
        stats = compute_statistics(df_mode, "metric")
        ref_stats = compute_statistics(ref_df_mode, "metric")

        if not stats or not ref_stats:
            continue

        # Plot
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        # Main Plot
        ax_main.plot(stats["centers"], stats["median"], label=run_label, marker="o")
        ax_main.fill_between(
            stats["centers"], stats["lower"], stats["upper"], alpha=0.2
        )

        ax_main.plot(
            ref_stats["centers"],
            ref_stats["median"],
            label=ref_label,
            marker="s",
            linestyle="--",
        )
        ax_main.fill_between(
            ref_stats["centers"], ref_stats["lower"], ref_stats["upper"], alpha=0.1
        )

        ax_main.set_ylabel(ylabel)
        ax_main.set_title(f"{title_prefix} - {mode.capitalize()}")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        ax_main.set_yscale("log")

        # Ratio Plot
        # Interpolate ref to match sample bins if needed, but here bins are fixed
        ratio = np.array(stats["median"]) / np.array(ref_stats["median"])

        ax_ratio.plot(stats["centers"], ratio, marker="o")
        ax_ratio.axhline(1.0, color="gray", linestyle="--")
        ax_ratio.set_ylabel("Ratio to IceMix")
        ax_ratio.set_xlabel("log10(Energy [GeV])")
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.set_ylim(0.5, 1.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_{mode}.png"))
        plt.close()


def filter_data_by_mode(df, mode):
    """Helper to filter dataframe by mode using plot_utils logic."""
    if mode == "all":
        return df.copy()
    elif mode == "tracks":
        return df[(abs(df["pid"]) == 14) & (df["interaction_type"] == 1)].copy()
    elif mode == "cascades":
        return df[~((abs(df["pid"]) == 14) & (df["interaction_type"] == 1))].copy()
    return df.copy()


def find_latest_icemix_baseline(run_dir):
    """Find the latest flat-format IceMix prediction in the enclosing outputs dir."""
    base_dir = os.path.dirname(run_dir)
    candidates = glob.glob(
        os.path.join(base_dir, "IceMix_*_job-*", "predictions", "results.csv")
    )

    best = None
    for result_csv in candidates:
        candidate_run = os.path.basename(os.path.dirname(os.path.dirname(result_csv)))
        match = re.match(
            r"^IceMix_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_job-(?P<job_id>\d+)$",
            candidate_run,
        )
        if not match:
            continue

        job_id = int(match.group("job_id"))
        if best is None or job_id > best[0]:
            best = (job_id, result_csv)

    return best


def get_run_label_from_name(run_dir):
    run_name = os.path.basename(run_dir)
    match = re.match(
        r"^(?P<project>.*?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_job-\d+$",
        run_name,
    )
    if match:
        return match.group("project")
    return "IceMix (This Run)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-csv", required=True, help="Path to prediction results.csv"
    )
    parser.add_argument(
        "--run-dir", required=True, help="Path to run directory (for logs)"
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save plots")
    args = parser.parse_args()

    setup_matplotlib_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Plot Loss
    plot_loss(args.run_dir, args.output_dir)

    run_label = get_run_label_from_name(args.run_dir)
    config_path = os.path.join(args.run_dir, ".hydra", "config.yaml")
    if os.path.exists(config_path):
        try:
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config and "project_name" in config:
                    run_label = config["project_name"]
        except Exception as e:
            logger.warning(f"Could not load run label from config: {e}")

    # Load Data
    logger.info(f"Loading results from {args.results_csv}")
    df = pd.read_csv(args.results_csv)

    baseline = find_latest_icemix_baseline(args.run_dir)
    if baseline:
        baseline_job_id, baseline_csv = baseline
        ref_label = f"IceMix nu_tau Baseline (job {baseline_job_id})"
        logger.info(f"Loading IceMix baseline from {baseline_csv}")
        ref_df = pd.read_csv(baseline_csv)

        # 2. Angular Resolution
        plot_resolution(
            df,
            ref_df,
            lambda d: calculate_angular_difference(
                d["azimuth"],
                d["zenith"],
                d["dir_x_pred"],
                d["dir_y_pred"],
                d["dir_z_pred"],
            ),
            "Angular Error [deg]",
            "Angular Resolution",
            args.output_dir,
            "angular_res",
            run_label=run_label,
            ref_label=ref_label,
        )

        # 3. Vertex Resolution
        plot_resolution(
            df,
            ref_df,
            lambda d: calculate_vertex_distance(d),
            "Vertex Error [m]",
            "Vertex Resolution",
            args.output_dir,
            "vertex_res",
            run_label=run_label,
            ref_label=ref_label,
        )
    else:
        logger.error("Could not find latest IceMix baseline prediction.")


if __name__ == "__main__":
    main()
