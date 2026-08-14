"""Plot reconstruction degradation under forced token removal."""

import argparse
import os
import glob
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
from plot_reference import get_run_label, filter_data_by_mode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_prediction_dirs(base_dir):
    """Return run directories containing ``predictions/results.csv``.

        ``base_dir`` is searched recursively. Presence of resilience results is
        checked later by :func:`main`.
        """
    # This finds prediction locations. But we want directories that have resilience_results too
    files = glob.glob(
        os.path.join(base_dir, "**", "predictions", "results.csv"), recursive=True
    )
    return [os.path.dirname(os.path.dirname(f)) for f in files]


def plot_resilience_comparison(
    model_data,
    metric_func,
    ylabel,
    title_prefix,
    output_dir,
    file_prefix,
    baseline_label="0% Drop",
):
    """Write drop-rate metric and baseline-ratio plots for one model.

        ``model_data`` contains ``(drop label, DataFrame)`` pairs and
        ``metric_func`` maps each filtered table to per-event errors. The label
        ``ylabel`` and ``title_prefix`` configure axes/titles, while
        ``file_prefix`` configures output filenames; ``baseline_label``
        selects the ratio denominator. Three 300-dpi PNGs are written beneath
        ``output_dir``. The function returns ``None``.
        """
    modes = ["all", "tracks", "cascades"]

    for mode in modes:
        logger.info(f"Generating resilience plot {title_prefix} ({mode})...")

        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        baseline_stats = None
        for label, df in model_data:
            if label == baseline_label:
                df_mode = filter_data_by_mode(df, mode)
                if not df_mode.empty:
                    values = metric_func(df_mode)
                    df_mode["metric"] = values
                    baseline_stats = compute_statistics(df_mode, "metric")
                    break

        if not baseline_stats:
            logger.warning(
                f"Could not find baseline label '{baseline_label}' for mode '{mode}'. Ratio will not be plotted."
            )

        seen_labels = set()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c_idx = 0

        for label, df in model_data:
            df_mode = filter_data_by_mode(df, mode)
            if df_mode.empty:
                continue

            values = metric_func(df_mode)
            df_mode["metric"] = values
            stats = compute_statistics(df_mode, "metric")

            if stats:
                is_baseline = label == baseline_label

                color = "black" if is_baseline else colors[c_idx % len(colors)]
                linewidth = 3 if is_baseline else 2
                linestyle = "--" if is_baseline else "-"
                alpha = 1.0 if is_baseline else 0.7
                zorder = 10 if is_baseline else 1

                if not is_baseline:
                    c_idx += 1

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
                    zorder=zorder,
                )

                ax_main.fill_between(
                    stats["centers"],
                    stats["lower"],
                    stats["upper"],
                    alpha=0.1,
                    color=color,
                    zorder=zorder,
                )

                if baseline_stats is not None:
                    ratio = np.array(stats["median"]) / np.array(
                        baseline_stats["median"]
                    )
                    ax_ratio.plot(
                        stats["centers"],
                        ratio,
                        marker="o",
                        alpha=alpha,
                        color=color,
                        linestyle=linestyle,
                        zorder=zorder,
                    )

        ax_main.set_ylabel(ylabel)
        ax_main.set_title(f"Resilience Plot: {title_prefix} - {mode.capitalize()}")
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_yscale("log")

        ax_ratio.axhline(1.0, color="gray", linewidth=2, linestyle="--")
        ax_ratio.set_ylabel(f"Ratio to {baseline_label}")
        ax_ratio.set_xlabel("log10(Energy [GeV])")
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.set_ylim(
            0.5, 3.0
        )  # Expanded y-limit for potentially bad resilience drops

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"resilience_{file_prefix}_{mode}.png"), dpi=300
        )
        plt.close()


def main():
    """Load baseline/drop CSV files and write per-run resilience plots.

        The command parses ``--base-dir`` and ``--output-dir``, creates per-run
        directories, and writes angular and vertex PNGs. Missing or unreadable
        inputs are logged and skipped.
        """
    parser = argparse.ArgumentParser(
        description="Generate resilience plots for each run."
    )
    parser.add_argument(
        "--base-dir",
        default="ice_mix/outputs",
        help="Base directory containing run outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="ice_mix/outputs/resilience_plots",
        help="Directory to save resilience plots",
    )
    args = parser.parse_args()

    setup_matplotlib_style()

    run_dirs = find_prediction_dirs(args.base_dir)
    logger.info(f"Found {len(run_dirs)} runs with predictions.")

    for run_dir in run_dirs:
        # Check if resilience results exist for this run
        resilience_dir = os.path.join(run_dir, "resilience_results")
        base_result_file = os.path.join(run_dir, "predictions", "results.csv")

        if not os.path.exists(resilience_dir) or not os.path.exists(base_result_file):
            continue

        resilience_files = glob.glob(os.path.join(resilience_dir, "drop_*.csv"))
        if not resilience_files:
            continue

        run_label = get_run_label(base_result_file)
        logger.info(f"Processing plots for run: {run_label} ({run_dir})")

        model_data = []

        # Load baseline
        try:
            base_df = pd.read_csv(base_result_file)
            model_data.append(("0% Drop", base_df))
        except Exception as e:
            logger.error(f"Error loading base predictions for {run_dir}: {e}")
            continue

        # Load drop files
        # Sort them basically by numeric drop percentage
        def extract_pct(filepath):
            """Extract the fractional token-drop value from a result filename.

            Parameters
            ----------
            filepath : str
                Path whose basename follows ``drop_<fraction>.csv``.

            Returns
            -------
            float
                Fraction encoded between the ``drop_`` prefix and ``.csv``.
            """
            filename = os.path.basename(filepath)
            return float(filename.replace("drop_", "").replace(".csv", ""))

        resilience_files = sorted(resilience_files, key=extract_pct)

        for rf in resilience_files:
            pct = extract_pct(rf)
            label = f"{pct*100:g}% Drop"
            try:
                df = pd.read_csv(rf)
                model_data.append((label, df))
            except Exception as e:
                logger.error(f"Error loading resilience result {rf}: {e}")

        if len(model_data) <= 1:
            logger.warning(f"No additional resilience files loaded for {run_dir}")
            continue

        safe_run_label = run_label.replace("/", "_").replace(" ", "_")
        output_dir = os.path.join(args.output_dir, safe_run_label)
        os.makedirs(output_dir, exist_ok=True)

        # Angular Resolution Plots
        plot_resilience_comparison(
            model_data,
            lambda d: calculate_angular_difference(
                d["azimuth"],
                d["zenith"],
                d["dir_x_pred"],
                d["dir_y_pred"],
                d["dir_z_pred"],
            ),
            "Angular Error [deg]",
            f"Angular Resolution Resilience ({run_label})",
            output_dir,
            "angular_res",
            baseline_label="0% Drop",
        )

        # Vertex Resolution Plots
        plot_resilience_comparison(
            model_data,
            lambda d: calculate_vertex_distance(d),
            "Vertex Error [m]",
            f"Vertex Resolution Resilience ({run_label})",
            output_dir,
            "vertex_res",
            baseline_label="0% Drop",
        )


if __name__ == "__main__":
    main()
