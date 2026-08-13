"""Compare IceMix prediction runs with reference reconstructions."""

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
    validate_matching_evaluation_manifests,
    validate_matching_events,
    write_plot_manifest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_result_files(base_dir):
    """Return the newest flat-format prediction CSV for each project.

    ``base_dir`` is searched recursively and the largest parsed Slurm job ID
    is retained for each exact project name.
    """
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
    """Parse ``result_path`` into ``(project, job_id)`` or return ``None``."""
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
    """Return the project name parsed from ``result_path`` or ``Unknown Run``."""
    parsed = parse_flat_run(result_path)
    return parsed[0] if parsed is not None else "Unknown Run"

def plot_reference_comparison(
    model_data, metric_func, ylabel, title_prefix, output_dir, file_prefix, baseline_label="IceMix"
):
    """Write metric and baseline-ratio plots for all topology modes.

    ``model_data`` contains ``(label, DataFrame)`` pairs; ``metric_func``
    returns one per-event error array. ``ylabel`` and ``title_prefix`` label
    the figures, while ``output_dir`` and ``file_prefix`` determine the three
    300-dpi PNG paths. ``baseline_label`` selects the ratio denominator. The
    function returns ``None`` and writes files as its side effect.
    """
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
    """Copy events selected as ``all``, charged-current tracks, or cascades.

    ``df`` must contain ``pid`` and ``interaction_type``. Unknown modes
    currently fall back to all events.
    """
    if mode == "all":
        return df.copy()
    elif mode == "tracks":
        return df[(abs(df["pid"]) == 14) & (df["interaction_type"] == 1)].copy()
    elif mode == "cascades":
        return df[~((abs(df["pid"]) == 14) & (df["interaction_type"] == 1))].copy()
    return df.copy()


def remove_stale_old_plots(output_dir):
    """Delete obsolete ``*_no_old_*.png`` products below ``output_dir``.

    Removal errors are logged and suppressed.
    """
    for path in glob.glob(os.path.join(output_dir, "*_no_old_*.png")):
        try:
            os.remove(path)
            logger.info("Removed stale OLD-filtered plot: %s", path)
        except OSError as e:
            logger.warning("Could not remove stale plot %s: %s", path, e)


def main():
    """Parse CLI paths, load predictions, and write reference PNG plots.

    The command creates ``--output-dir`` and removes obsolete plot products.
    Unreadable result tables are logged and skipped.
    """
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
    remove_stale_old_plots(args.output_dir)

    # 1. Find and Load Model Results. Only latest flat-format runs are eligible;
    # this makes IceMix resolve to the current nu_tau baseline run.
    result_files = find_result_files(args.base_dir)
    logger.info(f"Found {len(result_files)} latest result files.")

    # We sort to have somewhat deterministic behavior
    result_files = sorted(result_files)

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

    baseline_path = result_files[0]
    baseline_frame = pd.read_csv(baseline_path)
    for result_path in result_files[1:]:
        validate_matching_evaluation_manifests(baseline_path, result_path)
        validate_matching_events(
            baseline_frame,
            pd.read_csv(result_path),
            baseline_path,
            result_path,
        )
    write_plot_manifest(
        args.output_dir,
        "cross_project_reference_ratios",
        result_files,
    )

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


if __name__ == "__main__":
    main()
