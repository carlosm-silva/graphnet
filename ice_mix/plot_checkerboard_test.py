"""Plot the historical complementary-half pulse study.

Input directories and filenames retain the word ``checkerboard`` for
compatibility, but the evaluated halves are seeded random sequence-position
partitions rather than spatial detector cells.
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from plot_utils import setup_matplotlib_style
from plot_reference import filter_data_by_mode, get_run_label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_checkerboard_dirs(base_dir):
    """Return sorted run directories containing checkerboard predictions.

        ``base_dir`` is searched recursively for each run's
        ``checkerboard_results/checkerboard_half_1.csv`` marker file.
        """
    files = glob.glob(
        os.path.join(base_dir, "**", "checkerboard_results", "checkerboard_half_1.csv"),
        recursive=True,
    )
    return sorted([os.path.dirname(os.path.dirname(f)) for f in files])


def compute_custom_statistics(df, metric_col, x_col="energy", n_bins=20):
    """Bin a metric against log-energy or log-pulse count.

        Parameters
        ----------
        df : pandas.DataFrame
            Event table containing ``metric_col`` and ``x_col``.
        metric_col : str
            Column whose median and central 68-percent interval are calculated.
        x_col : {"energy", "n_pulses"}, default="energy"
            Quantity transformed with base-10 logarithm before binning.
        n_bins : int, default=20
            Number of equal-width bins between 1 and 4.

        Returns
        -------
        dict or None
            Bin centers, medians, 16th/84th percentiles, and counts, or ``None``
            for an empty input. Bins with at most five events contain NaNs.
        """
    if df.empty:
        return None

    df = df.copy()

    # Use log scale for energy and n_pulses
    if x_col == "energy":
        df["x_val"] = np.log10(df["energy"].clip(lower=10))  # Log10(Energy)
        bin_min, bin_max = 1.0, 4.0
    elif x_col == "n_pulses":
        df["x_val"] = np.log10(df["n_pulses"].clip(lower=10))  # Log10(n_pulses)
        bin_min, bin_max = 1.0, 4.0

    bins = np.linspace(bin_min, bin_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    stats = {"centers": centers, "median": [], "lower": [], "upper": [], "count": []}

    for i in range(len(bins) - 1):
        mask = (df["x_val"] >= bins[i]) & (df["x_val"] < bins[i + 1])
        vals = df.loc[mask, metric_col].dropna()

        if len(vals) > 5:
            stats["median"].append(np.median(vals))
            stats["lower"].append(np.percentile(vals, 16))
            stats["upper"].append(np.percentile(vals, 84))
            stats["count"].append(len(vals))
        else:
            stats["median"].append(np.nan)
            stats["lower"].append(np.nan)
            stats["upper"].append(np.nan)
            stats["count"].append(len(vals))

    return stats


def plot_checkerboard_metric(
    df_merged,
    metric_col,
    x_col,
    ylabel,
    xlabel,
    title_prefix,
    output_dir,
    file_prefix,
):
    """Write checkerboard stability plots for each topology selection.

        ``df_merged`` is the half-1/half-2 table produced by an inner merge on
        ``event_no`` alone. The caller does not validate database-qualified
        uniqueness, so the table is not proof of one-to-one event matching.
        ``metric_col`` is plotted against ``x_col`` using ``ylabel``, ``xlabel``, and
        ``title_prefix``. Three
        300-dpi PNG files are written below ``output_dir`` using ``file_prefix``;
        empty selections are skipped. The function returns ``None``.
        """
    modes = ["all", "tracks", "cascades"]

    for mode in modes:
        logger.info(
            f"Generating checkerboard plot {title_prefix} vs {x_col} ({mode})..."
        )

        # filter_data_by_mode accepts df directly. Ensure required cols exist.
        df_mode = filter_data_by_mode(df_merged, mode)
        if df_mode.empty:
            continue

        stats = compute_custom_statistics(df_mode, metric_col, x_col=x_col)
        if not stats:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(
            stats["centers"],
            stats["median"],
            marker="o",
            color="black",
            linewidth=2,
            label="Median",
        )
        ax.fill_between(
            stats["centers"],
            stats["lower"],
            stats["upper"],
            alpha=0.2,
            color="black",
            label="68% CI",
        )

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(f"Checkerboard {title_prefix} - {mode.capitalize()}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        # Use log scale if appropriate
        if "Separation" in title_prefix or "Error" in title_prefix:
            ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f"checkerboard_{file_prefix}_vs_{x_col}_{mode}.png"
            ),
            dpi=300,
        )
        plt.close()


def main():
    """Merge checkerboard halves and write angular/vertex stability plots.

        The command parses ``--base-dir`` and ``--output-dir``, reads CSV files,
        creates per-run output directories, and writes PNG files. Input failures
        are logged and skipped.
        """
    parser = argparse.ArgumentParser(
        description="Generate checkerboard plots for each run."
    )
    parser.add_argument(
        "--base-dir",
        default="ice_mix/outputs",
        help="Base directory containing run outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="ice_mix/outputs/checkerboard_plots",
        help="Directory to save checkerboard plots",
    )
    args = parser.parse_args()

    setup_matplotlib_style()

    run_dirs = find_checkerboard_dirs(args.base_dir)
    logger.info(f"Found {len(run_dirs)} runs with checkerboard predictions.")

    for run_dir in run_dirs:
        resilience_dir = os.path.join(run_dir, "checkerboard_results")
        base_result_file = os.path.join(run_dir, "predictions", "results.csv")

        file1 = os.path.join(resilience_dir, "checkerboard_half_1.csv")
        file2 = os.path.join(resilience_dir, "checkerboard_half_2.csv")

        if not os.path.exists(file1) or not os.path.exists(file2):
            logger.warning(f"Missing half 1 or half 2 predictions in {resilience_dir}")
            continue

        if os.path.exists(base_result_file):
            run_label = get_run_label(base_result_file)
        else:
            run_label = os.path.basename(run_dir)

        logger.info(f"Processing plots for run: {run_label} ({run_dir})")

        try:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
        except Exception as e:
            logger.error(f"Error loading CSV files in {resilience_dir}: {e}")
            continue

        # Merge the two dataframes on event_no.
        # Make sure they represent exactly the same events
        df_merged = pd.merge(df1, df2, on="event_no", suffixes=("_half1", "_half2"))

        # We also need energy, pid, n_pulses, interaction_type to be preserved for filtering. They are naturally the same in both halves.
        if "energy_half1" in df_merged.columns:
            df_merged["energy"] = df_merged["energy_half1"]
            df_merged["pid"] = df_merged["pid_half1"]
            df_merged["interaction_type"] = df_merged["interaction_type_half1"]
            df_merged["n_pulses"] = df_merged["n_pulses_half1"]

        # Optional: True labels as well (if needed for filtering but mode filtering should use the ones above)

        # Calculate Angular Separation (dot product -> angle in degrees)
        # Assuming predictions are unit vectors
        dot_product = (
            df_merged["dir_x_pred_half1"] * df_merged["dir_x_pred_half2"]
            + df_merged["dir_y_pred_half1"] * df_merged["dir_y_pred_half2"]
            + df_merged["dir_z_pred_half1"] * df_merged["dir_z_pred_half2"]
        )
        dot_product = np.clip(dot_product, -1.0, 1.0)
        df_merged["angular_separation"] = np.degrees(np.arccos(dot_product))

        # Calculate Vertex Separation (Euclidean distance in meters)
        df_merged["vertex_separation"] = np.sqrt(
            (df_merged["pos_x_pred_half1"] - df_merged["pos_x_pred_half2"]) ** 2
            + (df_merged["pos_y_pred_half1"] - df_merged["pos_y_pred_half2"]) ** 2
            + (df_merged["pos_z_pred_half1"] - df_merged["pos_z_pred_half2"]) ** 2
        )

        safe_run_label = run_label.replace("/", "_").replace(" ", "_")
        output_dir = os.path.join(args.output_dir, safe_run_label)
        os.makedirs(output_dir, exist_ok=True)

        for x_axis in ["energy", "n_pulses"]:
            x_label = f"log10({x_axis})" if x_axis == "energy" else "log10(N pulses)"
            plot_checkerboard_metric(
                df_merged,
                metric_col="angular_separation",
                x_col=x_axis,
                ylabel="Angular Separation [deg]",
                xlabel=x_label,
                title_prefix="Angular Separation",
                output_dir=output_dir,
                file_prefix="angular_sep",
            )
            plot_checkerboard_metric(
                df_merged,
                metric_col="vertex_separation",
                x_col=x_axis,
                ylabel="Vertex Separation [m]",
                xlabel=x_label,
                title_prefix="Vertex Separation",
                output_dir=output_dir,
                file_prefix="vertex_sep",
            )


if __name__ == "__main__":
    main()
