"""
Vertex Distance vs Number of Pulses (log-binned) Analysis and Plotting Tool

This module mirrors the energy-binned vertex analysis but bins by the number
of pulses (column `n_pulses`) and plots vertex distance as a function of
number of pulses on a logarithmic x-axis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 20
TITLE_FONT_SIZE = 24
FONT_FAMILY = "Times New Roman"

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "legend.fontsize": LABEL_FONT_SIZE,
})

COLORBLIND_FRIENDLY_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    "#8C4356", "#396A83", "#A87C4F", "#7FB80E", "#E17C05", "#66A5AD", "#B35C44",
    "#4A6A92", "#C77EB5", "#9B4B36", "#6C6F7C", "#008792", "#B89470"
]

# Default analysis parameters for pulses
DEFAULT_PULSES_BINS = 50
DEFAULT_MIN_EVENTS_PER_BIN = 0


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def calculate_vertex_distance(data: pd.DataFrame, pred_columns: List[str]) -> pd.DataFrame:
    """
    Calculate 3D Euclidean distance between predicted and true vertex positions.
    Required columns: 'position_x', 'position_y', 'position_z' and pred_columns [x,y,z].
    """
    if len(pred_columns) != 3:
        raise ValueError("pred_columns must contain exactly 3 elements [x, y, z]")

    required_cols = ['position_x', 'position_y', 'position_z'] + pred_columns
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    distances = np.sqrt(
        (data[pred_columns[0]] - data['position_x'])**2 +
        (data[pred_columns[1]] - data['position_y'])**2 +
        (data[pred_columns[2]] - data['position_z'])**2
    )
    data_copy = data.copy()
    data_copy['distance'] = distances
    logger.info(f"Calculated vertex distances for {len(data_copy)} events")
    return data_copy


def apply_event_filters(data: pd.DataFrame, filter_tracks: bool = False, filter_cascades: bool = False) -> pd.DataFrame:
    """
    Apply event selection filters to the dataset.
    - filter_tracks: pid=±14 and interaction_type==1
    - filter_cascades: everything except muon neutrino charged current
    """
    if not filter_tracks and not filter_cascades:
        return data

    initial_count = len(data)

    if filter_tracks:
        filtered_data = data[(np.abs(data['pid']) == 14) & (data['interaction_type'] == 1)].copy()
        filter_type = "track filters"
    else:
        filtered_data = data[~((np.abs(data['pid']) == 14) & (data['interaction_type'] == 1))].copy()
        filter_type = "cascade filters"

    final_count = len(filtered_data)
    logger.info(f"Applied {filter_type}: {initial_count} → {final_count} events ({100 * final_count / max(initial_count,1):.1f}% retained)")
    return filtered_data


def compute_pulse_binned_statistics(
    data: pd.DataFrame,
    num_bins: int = DEFAULT_PULSES_BINS,
    min_events: int = DEFAULT_MIN_EVENTS_PER_BIN,
    min_pulses: int = 1,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute distance statistics binned by number of pulses on a logarithmic scale.

    - Filters out rows with n_pulses < min_pulses (default: 1) to avoid log(0)
    - Uses log-spaced bins between min and max n_pulses present
    - Returns bin centers in linear pulse-count space for plotting on a log x-axis
    """
    if 'n_pulses' not in data.columns:
        raise KeyError("Missing required column 'n_pulses'")

    data_copy = data.copy()
    data_copy = data_copy[data_copy['n_pulses'] >= min_pulses]
    if len(data_copy) == 0:
        logger.warning("No events remaining after n_pulses filtering")
        return np.array([]), pd.DataFrame(columns=['Median', 'Count', '68% Lower', '68% Upper'])

    # Work in log10 space to create evenly spaced bins in log
    log_pulses = np.log10(data_copy['n_pulses'].astype(float))
    # Drop non-finite values just in case
    finite_mask = np.isfinite(log_pulses.values)
    if not np.all(finite_mask):
        data_copy = data_copy.iloc[finite_mask]
        log_pulses = log_pulses.iloc[finite_mask]

    log_min = float(np.floor(log_pulses.min()))
    log_max = float(np.ceil(log_pulses.max()))
    if not np.isfinite(log_min) or not np.isfinite(log_max):
        return np.array([]), pd.DataFrame(columns=['Median', 'Count', '68% Lower', '68% Upper'])

    log_bins = np.linspace(log_min, log_max, num=num_bins + 1)
    categories = pd.cut(log_pulses, bins=log_bins, include_lowest=True)

    # observed=True avoids creating empty groups for unobserved categories
    grouped = data_copy.groupby(categories, observed=True)

    def pct(x: pd.Series, q: float) -> float:
        arr = np.asarray(x)
        if arr.size == 0:
            return np.nan
        return float(np.percentile(arr, q))

    statistics = grouped['distance'].agg([
        'median',
        'count',
        lambda x: pct(x, 16),
        lambda x: pct(x, 84),
    ])
    statistics.columns = ['Median', 'Count', '68% Lower', '68% Upper']

    # Filter by min events
    statistics_filtered = statistics[statistics['Count'] >= min_events]

    if len(statistics_filtered) == 0:
        return np.array([]), statistics_filtered

    # Compute bin centers from interval index of the filtered statistics
    intervals = statistics_filtered.index.categories if hasattr(statistics_filtered.index, 'categories') else statistics_filtered.index
    # Align only to the observed/filtered intervals present in the index
    centers_log = []
    for interval in statistics_filtered.index:
        # interval is a pandas.Interval
        centers_log.append(0.5 * (interval.left + interval.right))
    bin_centers_filtered = 10 ** np.array(centers_log)

    logger.info(f"Computed statistics for {len(statistics_filtered)} pulse bins (filtered from {len(statistics)} total bins)")
    return bin_centers_filtered, statistics_filtered


def process_csv_file(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single CSV file according to the provided configuration and return
    pulse-binned statistics for vertex distance.
    """
    csv_filepath = Path(config['csv_filepath'])
    if not csv_filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_filepath}")

    logger.info(f"Processing file: {csv_filepath}")
    data = pd.read_csv(csv_filepath)
    logger.info(f"Loaded {len(data)} events from {csv_filepath.name}")

    if config.get('filter_tracks', False) or config.get('filter_cascades', False):
        data = apply_event_filters(
            data,
            filter_tracks=config.get('filter_tracks', False),
            filter_cascades=config.get('filter_cascades', False),
        )

    data = calculate_vertex_distance(data, config['pred_columns'])
    bin_centers, statistics = compute_pulse_binned_statistics(data)

    return {
        'label': config['label'],
        'bin_centers': bin_centers,
        'statistics': statistics,
        'show_in_ratio_plot': config.get('show_in_ratio_plot', True),
        'total_events': len(data),
    }


def process_multiple_csv_files(csv_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not csv_configs:
        logger.warning("No CSV configurations provided")
        return []

    processed_data: List[Dict[str, Any]] = []
    for i, config in enumerate(csv_configs):
        try:
            result = process_csv_file(config)
            processed_data.append(result)
            logger.info(f"Successfully processed file {i+1}/{len(csv_configs)}: {config['label']}")
        except Exception as e:
            logger.error(f"Failed to process file {i+1}/{len(csv_configs)}: {e}")
            continue

    logger.info(f"Successfully processed {len(processed_data)}/{len(csv_configs)} files")
    return processed_data


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def create_combined_plot_vs_pulses(
    processed_data: List[Dict[str, Any]],
    title: str,
    reference_label: str = "Reference",
    figure_size: Tuple[float, float] = (12, 10),
    dpi: int = 600,
    save_png: Optional[str] = None,
    save_pdf: Optional[str] = None,
    include_ratio_plot: bool = True,
) -> None:
    if not processed_data:
        logger.warning("No data provided for plotting")
        return

    ax_ratio: Optional[Axes] = None
    if include_ratio_plot:
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1,
            figsize=figure_size,
            dpi=dpi,
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True,
        )
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=figure_size, dpi=dpi)

    reference_median = None

    # Determine global x-limits from available bin centers
    all_x_values: List[float] = []

    for i, data in enumerate(processed_data):
        color = COLORBLIND_FRIENDLY_COLORS[i % len(COLORBLIND_FRIENDLY_COLORS)]

        x = np.asarray(data['bin_centers'])  # linear scale pulse counts
        y = data['statistics']['Median']
        y_lower = data['statistics']['68% Lower']
        y_upper = data['statistics']['68% Upper']

        all_x_values.extend(list(x))

        ax_main.plot(x, y, marker='o', linestyle='-', label=data['label'], color=color, linewidth=2)
        ax_main.fill_between(x, y_lower, y_upper, color=color, alpha=0.2)

        if data['label'] == reference_label:
            reference_median = np.array(y)
            logger.info(f"Using '{reference_label}' as reference for ratio plots")

    if include_ratio_plot and reference_median is not None and ax_ratio is not None:
        for i, data in enumerate(processed_data):
            if data['show_in_ratio_plot'] and data['label'] != reference_label:
                color = COLORBLIND_FRIENDLY_COLORS[i % len(COLORBLIND_FRIENDLY_COLORS)]
                y = np.array(data['statistics']['Median'])
                x = np.asarray(data['bin_centers'])

                valid_ref = reference_median > 0
                if np.any(valid_ref):
                    ratio = np.divide(y, reference_median, out=np.full_like(y, np.nan), where=valid_ref)
                    ax_ratio.plot(x[valid_ref], ratio[valid_ref], marker='o', linestyle='-', color=color, linewidth=2)

    # Configure main plot
    ax_main.set_xscale('log')
    if len(all_x_values) > 0:
        x_min = max(1.0, float(np.nanmin(all_x_values)))
        x_max = float(np.nanmax(all_x_values))
        if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
            ax_main.set_xlim(x_min, x_max)

    ax_main.set_ylim(0, 105)
    ax_main.set_ylabel("Vertex Distance (m)")
    ax_main.set_title(title)

    if include_ratio_plot:
        ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08))
    else:
        ax_main.legend(loc='best')
        ax_main.set_xlabel('Number of Pulses')

    ax_main.grid(True, alpha=0.3, which='both')

    if include_ratio_plot and ax_ratio is not None:
        ax_ratio.set_xscale('log')
        ax_ratio.set_ylim(0.9, 1.2)
        ax_ratio.set_yticks([0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
        ax_ratio.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        ax_ratio.set_ylabel('Ratio over Reference')
        ax_ratio.set_xlabel('Number of Pulses')
        ax_ratio.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_png:
        try:
            plt.savefig(save_png, dpi=dpi, bbox_inches='tight', format='png')
            logger.info(f"Plot saved as PNG: {save_png}")
        except Exception as e:
            logger.error(f"Failed to save PNG plot: {e}")

    if save_pdf:
        try:
            plt.savefig(save_pdf, dpi=dpi, bbox_inches='tight', format='pdf')
            logger.info(f"Plot saved as PDF: {save_pdf}")
        except Exception as e:
            logger.error(f"Failed to save PDF plot: {e}")

    plt.show()
    logger.info("Plot created successfully")


# =============================================================================
# CONFIGURATION AND MAIN EXECUTION
# =============================================================================

def main():
    """
    Configure your CSV files in the csv_configs list below, then run this script.
    This replicates the four analysis variations from the energy-based script.
    """

    analysis_configs = [
        {
            "name": "numu_tracks",
            "title": "Vertex Distance vs Number of Pulses (νμ Tracks Only)",
            "reference_label": r"TANGO $\alpha=0.04$ (Reference)",
            "filename": "vertex_pulses_numu_tracks.png",
            "csv_configs": [
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0$ (Sequence Length 80)",
                    "filter_tracks": True,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.026_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.026$",
                    "filter_tracks": True,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.040_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.040$",
                    "filter_tracks": True,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.060_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.060$",
                    "filter_tracks": True,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW_with_n_pulses.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"TANGO $\alpha=0.04$ (Reference)",
                    "filter_tracks": True,
                    "filter_cascades": False,
                    "show_in_ratio_plot": False,
                },
            ],
        },
        {
            "name": "numu_cascades",
            "title": "Vertex Distance vs Number of Pulses (νμ Cascades Only)",
            "reference_label": r"TANGO $\alpha=0.04$ (Reference)",
            "filename": "vertex_pulses_numu_cascades.png",
            "csv_configs": [
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0$ (Sequence Length 80)",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.026_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.026$",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.040_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.040$",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.060_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.060$",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW_with_n_pulses.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"TANGO $\alpha=0.04$ (Reference)",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": False,
                },
            ],
        },
        {
            "name": "nue_cascades",
            "title": "Vertex Distance vs Number of Pulses (νe Cascades Only)",
            "reference_label": r"TANGO $\alpha=0.04$ (Reference)",
            "filename": "vertex_pulses_nue_cascades.png",
            "csv_configs": [
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0$ (Sequence Length 80)",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.026_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.026$",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.040_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.040$",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.060_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.060$",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW_with_n_pulses.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"TANGO $\alpha=0.04$ (Reference)",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": False,
                },
            ],
        },
        {
            "name": "all_events",
            "title": "Vertex Distance vs Number of Pulses (All Events)",
            "reference_label": r"TANGO $\alpha=0.04$ (Reference)",
            "filename": "vertex_pulses_all_events.png",
            "csv_configs": [
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0$ (Sequence Length 80)",
                    "filter_tracks": False,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.026_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.026$",
                    "filter_tracks": False,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.040_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.040$",
                    "filter_tracks": False,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.060_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.060$",
                    "filter_tracks": False,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True,
                },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW_with_n_pulses.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"TANGO $\alpha=0.04$ (Reference)",
                    "filter_tracks": False,
                    "filter_cascades": False,
                    "show_in_ratio_plot": False,
                },
            ],
        },
    ]

    base_path = 'carlos_tests/icemix_tiny/plots'

    try:
        for config in analysis_configs:
            logger.info(f"Starting vertex pulses analysis for: {config['name']}")

            processed_statistics = process_multiple_csv_files(config['csv_configs'])
            if not processed_statistics:
                logger.error(f"No data was successfully processed for {config['name']}.")
                continue

            save_png_path = os.path.join(base_path, config['filename'])

            create_combined_plot_vs_pulses(
                processed_statistics,
                title=config['title'],
                reference_label=config['reference_label'],
                save_png=save_png_path,
                save_pdf=None,
                include_ratio_plot=True,
            )

            logger.info(f"Completed analysis for: {config['name']}")

        logger.info("All vertex distance vs pulses analyses completed successfully!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()


