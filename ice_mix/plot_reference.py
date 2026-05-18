import argparse
import os
import glob
import pandas as pd
import numpy as np
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

def find_result_files(base_dir):
    """Scan directory for results.csv files."""
    files = glob.glob(
        os.path.join(base_dir, "**", "predictions", "results.csv"), recursive=True
    )
    return files

def get_run_label(result_path):
    """Generate a label for the run based on config.yaml or directory name."""
    import re
    import yaml

    run_dir = os.path.dirname(os.path.dirname(result_path))
    dir_name = os.path.basename(run_dir)

    # 1. Try to extract from the new flat directory format:
    # Pattern: {project_name}_{YYYY-MM-DD}_{HH-MM-SS}_job-{job_id}
    match = re.search(r"^(.*?)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", dir_name)
    if match:
        return match.group(1)

    # 2. Old nested format fallback (YYYY-MM-DD/HH-MM-SS with .hydra config)
    try:
        config_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config and "project_name" in config:
                    proj_name = config["project_name"]
                    if proj_name == "IceMix":
                        return proj_name
                    # Mark as OLD to avoid confusion with new runs of the same name
                    return f"{proj_name} (OLD)"
    except Exception as e:
        logger.warning(f"Could not load run label from config: {e}")

    # 3. Ultimate Fallback to path string
    parts = result_path.split(os.sep)
    if len(parts) >= 4:
        return f"{parts[-4]}/{parts[-3]}"
    return "Unknown Run"

def plot_reference_comparison(
    model_data, metric_func, ylabel, title_prefix, output_dir, file_prefix, baseline_label="IceMix"
):
    """Plot reference comparison for all models relative to baseline."""
    modes = ["all", "tracks", "cascades"]

    for mode in modes:
        logger.info(f"Generating reference plot {title_prefix} ({mode})...")

        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        # 1. Find the baseline stats
        baseline_stats = None
        for label, df in model_data:
            if label == baseline_label:
                df_mode = filter_data_by_mode(df, mode)
                if not df_mode.empty:
                    values = metric_func(df_mode)
                    df_mode["metric"] = values
                    baseline_stats = compute_statistics(df_mode, "metric")
                    break  # Use the first one found

        if not baseline_stats:
            logger.warning(f"Could not find baseline label '{baseline_label}' for mode '{mode}'. Ratio will not be plotted.")

        # Tracking seen labels for the legend
        seen_labels = set()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        c_idx = 0

        # 2. Plot models
        for label, df in model_data:
            df_mode = filter_data_by_mode(df, mode)
            if df_mode.empty:
                continue

            values = metric_func(df_mode)
            df_mode["metric"] = values
            stats = compute_statistics(df_mode, "metric")

            if stats:
                is_baseline = (label == baseline_label)
                
                color = "black" if is_baseline else colors[c_idx % len(colors)]
                linewidth = 3 if is_baseline else 2
                linestyle = "--" if is_baseline else "-"
                alpha = 1.0 if is_baseline else 0.7
                zorder = 10 if is_baseline else 1

                if not is_baseline:
                    c_idx += 1

                # Avoid duplicate labels in legend
                plot_label = label if label not in seen_labels else None
                seen_labels.add(label)
                
                ax_main.plot(
                    stats["centers"],
                    stats["median"],
                    label=plot_label,
                    marker="o",
                    alpha=alpha,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=zorder
                )
                
                # We don't need a label for the fill_between
                ax_main.fill_between(
                    stats["centers"], stats["lower"], stats["upper"], alpha=0.1, color=color, zorder=zorder
                )

                if baseline_stats is not None:
                    ratio = np.array(stats["median"]) / np.array(baseline_stats["median"])
                    # Use label in ratio plot only if it hasn't been added yet (actually we don't use legend here)
                    ax_ratio.plot(
                        stats["centers"],
                        ratio,
                        marker="o",
                        alpha=alpha,
                        color=color,
                        linestyle=linestyle,
                        zorder=zorder
                    )

        ax_main.set_ylabel(ylabel)
        ax_main.set_title(f"Reference Plot: {title_prefix} - {mode.capitalize()}")
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_yscale("log")

        ax_ratio.axhline(1.0, color="gray", linewidth=2, linestyle="--")
        ax_ratio.set_ylabel(f"Ratio to {baseline_label}")
        ax_ratio.set_xlabel("log10(Energy [GeV])")
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.set_ylim(0.5, 1.5)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"reference_{file_prefix}_{mode}.png"), dpi=300
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
        "--output-dir", required=True, help="Directory to save reference plots"
    )
    args = parser.parse_args()

    setup_matplotlib_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Find and Load Model Results (Do NOT load TANGO Ref)
    result_files = find_result_files(args.base_dir)
    logger.info(f"Found {len(result_files)} result files.")

    # We sort to have somewhat deterministic behavior
    result_files = sorted(result_files)

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

    has_old = any("(OLD)" in label for label, _ in model_data)

    # 2. Angular Resolution Reference Plots
    plot_reference_comparison(
        model_data,
        lambda d: calculate_angular_difference(
            d["azimuth"], d["zenith"], d["dir_x_pred"], d["dir_y_pred"], d["dir_z_pred"]
        ),
        "Angular Error [deg]",
        "Angular Resolution",
        args.output_dir,
        "angular_res",
        baseline_label="IceMix"
    )
    if has_old:
        model_data_no_old = [(label, df) for label, df in model_data if "(OLD)" not in label]
        if model_data_no_old:
            plot_reference_comparison(
                model_data_no_old,
                lambda d: calculate_angular_difference(
                    d["azimuth"], d["zenith"], d["dir_x_pred"], d["dir_y_pred"], d["dir_z_pred"]
                ),
                "Angular Error [deg]",
                "Angular Resolution (No OLD)",
                args.output_dir,
                "angular_res_no_old",
                baseline_label="IceMix"
            )

    # 3. Vertex Resolution Reference Plots
    plot_reference_comparison(
        model_data,
        lambda d: calculate_vertex_distance(d),
        "Vertex Error [m]",
        "Vertex Resolution",
        args.output_dir,
        "vertex_res",
        baseline_label="IceMix"
    )
    if has_old:
        if model_data_no_old:
            plot_reference_comparison(
                model_data_no_old,
                lambda d: calculate_vertex_distance(d),
                "Vertex Error [m]",
                "Vertex Resolution (No OLD)",
                args.output_dir,
                "vertex_res_no_old",
                baseline_label="IceMix"
            )


if __name__ == "__main__":
    main()
