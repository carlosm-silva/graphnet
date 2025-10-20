"""
Angular Resolution vs Number of Pulses Analysis

This module mirrors the functionality of `angular_plots.py`, but instead of
binning by neutrino energy, it bins by the number of pulses (`n_pulses`) found
in the event CSVs.
"""

from typing import List, Tuple, Dict, Any, Optional, Union

import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Colorblind-friendly color palette
COLORBLIND_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
    "#D55E00", "#CC79A7", "#8C4356", "#396A83", "#A87C4F",
    "#7FB80E", "#E17C05", "#66A5AD", "#B35C44", "#4A6A92",
    "#C77EB5", "#9B4B36", "#6C6F7C", "#008792", "#B89470",
]

# Font and plot styling constants
PREFERRED_FONTS = [
    "Times New Roman",  # Preferred font
    "Liberation Serif", # Good alternative (often pre-installed on Linux)
    "DejaVu Serif",     # Common fallback
    "serif"             # Generic serif fallback
]
TITLE_FONT_SIZE = 26
LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 20
FIGURE_DPI = 600
FIGURE_SIZE = (14 * 1.5, 8 * 1.5)

# Analysis constants
ELECTRON_NEUTRINO_PID = 12
MUON_NEUTRINO_PID = 14
TAU_NEUTRINO_PID = 16
CHARGED_CURRENT_INTERACTION = 1
DEFAULT_PULSE_BINS = 20
ANGLE_PLOT_MAX = 60.0


# Font detection and configuration
def get_available_font() -> str:
    """
    Detect which font from the preferred list is available on the system.

    Returns:
    --------
    str
        The name of the first available font from PREFERRED_FONTS
    """
    import matplotlib.font_manager as fm

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in PREFERRED_FONTS:
        if font in available_fonts or font == "serif":
            logger.info(f"Using font: {font}")
            return font

    # Ultimate fallback
    logger.warning("No preferred fonts found, using default matplotlib font")
    return "DejaVu Sans"


def setup_matplotlib_style() -> None:
    """Configure matplotlib with consistent styling."""
    font_family = get_available_font()

    mpl.rcParams.update({
        'font.family': font_family,
        'mathtext.fontset': 'custom',
        'mathtext.rm': font_family,
        'font.size': LABEL_FONT_SIZE,
    })

    plt.rcParams.update({
        "font.size": LABEL_FONT_SIZE,
        "font.family": font_family,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "legend.fontsize": LABEL_FONT_SIZE,
    })


# ============================================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================================

def compute_log_bin_edges_from_data(values: Union[np.ndarray, pd.Series], num_bins: int) -> np.ndarray:
    """
    Compute log-spaced bin edges from data, handling zeros by ignoring them in the
    log-range determination. Zeros will be included in the first bin during binning.
    """
    arr = np.asarray(values)
    positive = arr[(arr > 0) & np.isfinite(arr)]
    if positive.size == 0:
        # Fallback: arbitrary small positive range
        return np.logspace(0, 1, num_bins + 1)
    min_val = float(np.nanmin(positive))
    max_val = float(np.nanmax(positive))
    if min_val <= 0 or not np.isfinite(min_val):
        min_val = 1.0
    if not np.isfinite(max_val):
        max_val = max(min_val * 10.0, 10.0)
    if min_val == max_val:
        # Expand a narrow range multiplicatively
        min_val *= 0.8
        max_val *= 1.2
        if min_val <= 0:
            min_val = max_val / 10.0
    return np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1)

def calculate_angular_difference(
    true_azimuth: Union[np.ndarray, Any],
    true_zenith: Union[np.ndarray, Any],
    pred_x: Union[np.ndarray, Any],
    pred_y: Union[np.ndarray, Any],
    pred_z: Union[np.ndarray, Any],
) -> np.ndarray:
    """
    Calculate the angular difference between true and predicted neutrino directions.
    Uses the numerically stable haversine formula instead of arccos.

    Parameters:
    -----------
    true_azimuth : array-like
        True azimuth angles in radians
    true_zenith : array-like
        True zenith angles in radians
    pred_x, pred_y, pred_z : array-like
        Predicted direction components (normalized)

    Returns:
    --------
    np.ndarray
        Angular differences in degrees
    """
    true_azimuth = np.asarray(true_azimuth)
    true_zenith = np.asarray(true_zenith)
    pred_x = np.asarray(pred_x)
    pred_y = np.asarray(pred_y)
    pred_z = np.asarray(pred_z)

    pred_azimuth = np.arctan2(pred_y, pred_x)
    pred_zenith = np.arctan2(np.sqrt(pred_x ** 2 + pred_y ** 2), pred_z)

    delta_azimuth = pred_azimuth - true_azimuth
    delta_zenith = pred_zenith - true_zenith

    haversine = (
        np.sin(delta_zenith / 2.0) ** 2
        + np.sin(true_zenith) * np.sin(pred_zenith) * np.sin(delta_azimuth / 2.0) ** 2
    )
    haversine = np.clip(haversine, 0.0, 1.0)

    angular_separation = 2.0 * np.arcsin(np.sqrt(haversine))
    return np.degrees(angular_separation)


def load_and_filter_data(
    csv_filepath: str,
    include_charged_current: bool = True,
    include_neutral_current: bool = True,
    neutrino_pid: int = ELECTRON_NEUTRINO_PID,
) -> pd.DataFrame:
    """
    Load CSV data and apply filters for neutrino type and interaction type.

    Returns a dataframe with: ['event_no', 'n_pulses', 'angle']
    """
    try:
        logger.info(f"Loading data from {csv_filepath}")
        df = pd.read_csv(csv_filepath)

        # Filter by neutrino type
        df = df[np.abs(df["pid"]) == neutrino_pid]
        logger.info(f"After neutrino type filter: {len(df)} events")

        # Apply interaction type filters
        if include_charged_current and not include_neutral_current:
            df = df[df["interaction_type"] == CHARGED_CURRENT_INTERACTION]
            logger.info("Keeping only charged current interactions")
        elif include_neutral_current and not include_charged_current:
            df = df[df["interaction_type"] != CHARGED_CURRENT_INTERACTION]
            logger.info("Keeping only neutral current interactions")
        elif not include_charged_current and not include_neutral_current:
            logger.warning("No interaction types selected, returning empty dataframe")
            return pd.DataFrame(columns=["event_no", "n_pulses", "angle"])
        else:
            logger.info("Keeping both charged and neutral current interactions")

        logger.info(f"After interaction type filter: {len(df)} events")

        # Calculate angular differences
        df["angle"] = calculate_angular_difference(
            df["azimuth"], df["zenith"], df["dir_x_pred"], df["dir_y_pred"], df["dir_z_pred"]
        )

        # Ensure n_pulses exists
        if "n_pulses" not in df.columns:
            raise KeyError("Column 'n_pulses' not found in CSV")

        return df[["event_no", "n_pulses", "angle"]]

    except FileNotFoundError:
        logger.error(f"File not found: {csv_filepath}")
        raise
    except KeyError as e:
        logger.error(f"Required column missing from CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise


def compute_angular_statistics_by_pulses(
    df: pd.DataFrame,
    num_bins: int = DEFAULT_PULSE_BINS,
    bin_edges: Optional[np.ndarray] = None,
) -> Dict[str, List[float]]:
    """
    Compute angular resolution statistics binned by number of pulses.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing 'n_pulses' and 'angle' columns
    num_bins : int
        Number of bins to use if bin_edges not provided
    bin_edges : Optional[np.ndarray]
        Explicit bin edges to use (shared across models for consistent ratios)

    Returns:
    --------
    Dict[str, List[float]]
        Dictionary containing bin centers, median values, and 68% confidence intervals
    """
    df = df.copy()

    # Define pulse bins
    if bin_edges is None:
        pulses_min = float(np.nanmin(df["n_pulses"]))
        pulses_max = float(np.nanmax(df["n_pulses"]))
        if pulses_min == pulses_max:
            pulses_min = max(0.0, pulses_min - 0.5)
            pulses_max = pulses_max + 0.5
        bin_edges = np.linspace(pulses_min, pulses_max, num_bins + 1)

    # Use geometric mean for centers when bins are positive (for log-scale x)
    bin_centers_list: List[float] = []
    for i in range(len(bin_edges) - 1):
        a = float(bin_edges[i])
        b = float(bin_edges[i + 1])
        if a > 0 and b > 0:
            bin_centers_list.append(float(np.sqrt(a * b)))
        else:
            bin_centers_list.append(0.5 * (a + b))
    bin_centers = np.array(bin_centers_list)

    statistics: Dict[str, List[float]] = {
        "bin_centers": bin_centers.tolist(),
        "percentile_16": [],
        "median": [],
        "percentile_84": [],
    }

    # Calculate statistics for each bin
    for i in range(len(bin_edges) - 1):
        base_mask = (df["n_pulses"] >= bin_edges[i]) & (df["n_pulses"] < bin_edges[i + 1])
        if i == 0 and bin_edges[0] > 0:
            # Include zeros in the first bin when using positive log bins
            zero_mask = (df["n_pulses"] == 0) & (df["n_pulses"] < bin_edges[i + 1])
            bin_mask = base_mask | zero_mask
        else:
            bin_mask = base_mask
        bin_angles = df[bin_mask]["angle"]

        if len(bin_angles) > 0:
            statistics["median"].append(float(np.median(bin_angles)))
            statistics["percentile_16"].append(float(np.percentile(bin_angles, 16)))
            statistics["percentile_84"].append(float(np.percentile(bin_angles, 84)))
            logger.debug(
                f"Pulse bin {i}: {len(bin_angles)} events, median = {statistics['median'][-1]:.2f}°"
            )
        else:
            statistics["median"].append(np.nan)
            statistics["percentile_16"].append(np.nan)
            statistics["percentile_84"].append(np.nan)
            logger.warning(f"Pulse bin {i} is empty")

    return statistics


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_comparison_plot(
    model_results: List[Tuple[Dict, str, bool]],
    reference_stats: Dict[str, List[float]],
    title: str = r"GRECO $\nu_e$ Median Angular Resolution vs Number of Pulses",
    save_png: Optional[str] = None,
    save_pdf: Optional[str] = None,
    show_plot: bool = True,
    include_ratio_plot: bool = True,
    log_scale_y: bool = False,
    log_scale_x: bool = True,
) -> None:
    """
    Create a comparison plot showing angular resolution vs number of pulses for multiple models.
    """
    if include_ratio_plot:
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, figsize=FIGURE_SIZE, dpi=FIGURE_DPI, gridspec_kw={'height_ratios': [3, 1]}, sharex=True
        )
        _plot_ratio_comparison(ax_ratio, model_results, reference_stats, log_scale_x=log_scale_x)
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    _plot_main_comparison(ax_main, model_results, title, include_ratio_plot, log_scale_y, log_scale_x)

    plt.tight_layout()

    if save_png:
        logger.info(f"Saving plot as PNG: {save_png}")
        plt.savefig(save_png, format='png', dpi=FIGURE_DPI, bbox_inches='tight')

    if save_pdf:
        logger.info(f"Saving plot as PDF: {save_pdf}")
        plt.savefig(save_pdf, format='pdf', dpi=FIGURE_DPI, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


def _plot_main_comparison(
    ax: plt.Axes,
    model_results: List[Tuple[Dict, str, bool]],
    title: str,
    include_ratio_plot: bool = True,
    log_scale_y: bool = False,
    log_scale_x: bool = True,
) -> None:
    for i, (stats, label, _) in enumerate(model_results):
        color = COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)]

        ax.plot(
            stats["bin_centers"], stats["median"], marker='o', linestyle='-', color=color, label=label, linewidth=2, markersize=6
        )

        ax.fill_between(
            stats["bin_centers"], stats["percentile_16"], stats["percentile_84"], color=color, alpha=0.2
        )

    # Configure main plot
    # X-limits based on first model bin centers (all models share same bins)
    if model_results:
        x_min = np.nanmin(model_results[0][0]["bin_centers"]) if model_results[0][0]["bin_centers"] else 0
        x_max = np.nanmax(model_results[0][0]["bin_centers"]) if model_results[0][0]["bin_centers"] else 1
        ax.set_xlim(x_min, x_max)

    if log_scale_x:
        # Only set log scale if bins are strictly positive
        if model_results and np.all(np.array(model_results[0][0]["bin_centers"]) > 0):
            ax.set_xscale('log')

    if log_scale_y:
        ax.set_yscale('log')
        # Compute sensible lower bound for log scale
        min_values: List[float] = []
        for stats, _, _ in model_results:
            valid_vals = [v for v in stats["percentile_16"] if not np.isnan(v) and v > 0]
            if len(valid_vals) > 0:
                min_values.append(float(np.min(valid_vals)))
        if min_values:
            y_min = min(min_values) * 0.8
            ax.set_ylim(y_min, ANGLE_PLOT_MAX)
        else:
            ax.set_ylim(0.1, ANGLE_PLOT_MAX)

        from matplotlib.ticker import LogLocator
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.15, linewidth=0.5, axis='y')
    else:
        ax.set_ylim(0, ANGLE_PLOT_MAX)
        ax.grid(True, alpha=0.3)

    ax.set_ylabel("Angular Difference (°)")
    ax.set_title(title)

    if include_ratio_plot:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    else:
        ax.legend(loc="best")
        ax.set_xlabel("Number of Pulses")


def _plot_ratio_comparison(
    ax: plt.Axes,
    model_results: List[Tuple[Dict, str, bool]],
    reference_stats: Dict[str, List[float]],
    log_scale_x: bool = True,
) -> None:
    reference_median = np.array(reference_stats["median"])

    for i, (stats, label, show_ratio) in enumerate(model_results):
        if show_ratio:
            color = COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)]
            model_median = np.array(stats["median"])
            ratio = model_median / reference_median
            ax.plot(stats["bin_centers"], ratio, marker='o', linestyle='-', color=color, label=label, linewidth=2, markersize=6)

    ax.axhline(1, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel("Ratio over Reference")
    ax.set_xlabel("Number of Pulses")
    ax.grid(True, alpha=0.3)

    if log_scale_x:
        # Only set log scale if bins are strictly positive
        if model_results and np.all(np.array(model_results[0][0]["bin_centers"]) > 0):
            ax.set_xscale('log')


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution function for angular resolution analysis vs number of pulses.

    Generates four plot variations analogous to `angular_plots.py`.
    """
    setup_matplotlib_style()

    # ------------------------------------------------------------------------
    # CONFIGURATION SECTION
    # ------------------------------------------------------------------------

    # List of CSV files to analyze: (filepath, label, show_in_ratio_plot)
    csv_files: List[Tuple[str, str, bool]] = [
        (
            "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_example/results.csv",
            r"IceMix Tiny $\alpha=0$ (Sequence Length 80)",
            True,
        ),
        (
            "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.026_example/results.csv",
            r"IceMix Tiny $\alpha=0.026$",
            True,
        ),
        # (
        #     "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.040_example/results.csv",
        #     r"IceMix Tiny $\alpha=0.040$",
        #     True,
        # ),
        # (
        #     "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.060_example/results.csv",
        #     r"IceMix Tiny $\alpha=0.060$",
        #     True,
        # ),
        (
            "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW_with_n_pulses.csv",
            r"TANGO $\alpha=0.04$ (Reference)",
            False,
        ),
    ]

    # Reference CSV file for ratio calculations and defining shared pulse bins
    reference_csv_path = \
        "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW_with_n_pulses.csv"

    # Plot configuration options
    include_ratio_plot = True
    log_scale_y = True
    show_interactive_plot = False

    # Create plots directory
    base_path = 'carlos_tests/icemix_tiny/plots'
    os.makedirs(base_path, exist_ok=True)

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------
    if not csv_files:
        logger.warning("No CSV files configured. Please add file paths to csv_files list.")
        return
    if reference_csv_path is None:
        logger.warning("No reference CSV configured. Please set reference_csv_path.")
        return

    # ------------------------------------------------------------------------
    # GENERATE FOUR PLOT VARIATIONS
    # ------------------------------------------------------------------------
    analysis_configs = [
        {
            "name": "numu_tracks",
            "title": "GRECO νμ Median Angular Resolution vs Pulses (Tracks Only)",
            "include_charged_current": True,
            "include_neutral_current": False,
            "neutrino_type": MUON_NEUTRINO_PID,
            "filename": "angular_resolution_vs_pulses_numu_tracks.png",
        },
        {
            "name": "numu_cascades",
            "title": "GRECO νμ Median Angular Resolution vs Pulses (Cascades Only)",
            "include_charged_current": False,
            "include_neutral_current": True,
            "neutrino_type": MUON_NEUTRINO_PID,
            "filename": "angular_resolution_vs_pulses_numu_cascades.png",
        },
        {
            "name": "nue_cascades",
            "title": "GRECO νe Median Angular Resolution vs Pulses (Cascades Only)",
            "include_charged_current": False,
            "include_neutral_current": True,
            "neutrino_type": ELECTRON_NEUTRINO_PID,
            "filename": "angular_resolution_vs_pulses_nue_cascades.png",
        },
        {
            "name": "all_events",
            "title": "GRECO All Events Median Angular Resolution vs Pulses",
            "include_charged_current": True,
            "include_neutral_current": True,
            "neutrino_type": MUON_NEUTRINO_PID,
            "filename": "angular_resolution_vs_pulses_all_events.png",
        },
    ]

    try:
        for config in analysis_configs:
            logger.info(f"Generating plot: {config['name']}")

            # Prepare shared bins from reference to ensure consistent ratios
            logger.info(f"Computing shared pulse bins from reference: {reference_csv_path}")
            reference_df_for_bins = load_and_filter_data(
                reference_csv_path,
                config["include_charged_current"],
                config["include_neutral_current"],
                config["neutrino_type"],
            )
            if len(reference_df_for_bins) == 0:
                logger.error(f"No data found in reference file after filtering for {config['name']}")
                continue
            # Use logarithmic bins based on positive n_pulses in the reference
            shared_bins = compute_log_bin_edges_from_data(
                reference_df_for_bins["n_pulses"], DEFAULT_PULSE_BINS
            )
            # Guard against degenerate bins
            if (
                not np.all(np.isfinite(shared_bins))
                or np.nanmin(shared_bins) <= 0
                or np.nanmin(shared_bins) == np.nanmax(shared_bins)
            ):
                # Fallback to linear small range if something went wrong
                shared_bins = np.linspace(0.0, 1.0, DEFAULT_PULSE_BINS + 1)

            # Process model files
            model_results: List[Tuple[Dict, str, bool]] = []
            for filepath, label, show_ratio in csv_files:
                logger.info(f"Processing {label}: {filepath}")
                df = load_and_filter_data(
                    filepath,
                    config["include_charged_current"],
                    config["include_neutral_current"],
                    config["neutrino_type"],
                )
                if len(df) == 0:
                    logger.warning(f"No data found for {label} after filtering")
                    continue
                stats = compute_angular_statistics_by_pulses(df, bin_edges=shared_bins)
                model_results.append((stats, label, show_ratio))

            # Process reference file (use same bins)
            logger.info(f"Processing reference file: {reference_csv_path}")
            reference_df = reference_df_for_bins
            reference_stats = compute_angular_statistics_by_pulses(reference_df, bin_edges=shared_bins)

            if len(model_results) == 0:
                logger.warning(f"No model results to plot for {config['name']}")
                continue

            # Generate comparison plot
            save_png_path = os.path.join(base_path, config["filename"])
            create_comparison_plot(
                model_results,
                reference_stats,
                config["title"],
                save_png=save_png_path,
                save_pdf=None,
                show_plot=show_interactive_plot,
                include_ratio_plot=include_ratio_plot,
                log_scale_y=log_scale_y,
            )

            logger.info(f"Completed plot: {config['name']}")

        logger.info("All four plot variations completed successfully!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()


