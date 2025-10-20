"""
Vertex Distance Analysis and Plotting Tool

This module provides functionality for analyzing vertex reconstruction accuracy
from neutrino detection data. It processes CSV files containing predicted and
true vertex positions, calculates reconstruction errors, and generates
comparative plots with statistical analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import os

import pandas as pd
import numpy as np
import sqlite3
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

# Font configuration for publication-quality plots
LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 20
TITLE_FONT_SIZE = 24
FONT_FAMILY = "Times New Roman"

# Configure matplotlib for consistent styling
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

# Colorblind-friendly color palette for multiple data series
COLORBLIND_FRIENDLY_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    "#8C4356", "#396A83", "#A87C4F", "#7FB80E", "#E17C05", "#66A5AD", "#B35C44",
    "#4A6A92", "#C77EB5", "#9B4B36", "#6C6F7C", "#008792", "#B89470"
]

# Default analysis parameters
DEFAULT_ENERGY_BINS = 50
DEFAULT_ENERGY_RANGE = (1, 4)  # log10(GeV)
DEFAULT_MIN_EVENTS_PER_BIN = 0  # Minimum events required per energy bin

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def calculate_vertex_distance(data: pd.DataFrame, pred_columns: List[str]) -> pd.DataFrame:
    """
    Calculate 3D Euclidean distance between predicted and true vertex positions.
    
    Args:
        data: DataFrame containing both predicted and true vertex positions
        pred_columns: List of column names for predicted positions [x, y, z]
        
    Returns:
        DataFrame with added 'distance' column containing reconstruction errors
        
    Raises:
        KeyError: If required columns are missing from the DataFrame
        ValueError: If pred_columns doesn't contain exactly 3 elements
    """
    if len(pred_columns) != 3:
        raise ValueError("pred_columns must contain exactly 3 elements [x, y, z]")
    
    required_cols = ['position_x', 'position_y', 'position_z'] + pred_columns
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    try:
        # Calculate 3D Euclidean distance between predicted and true positions
        distances = np.sqrt(
            (data[pred_columns[0]] - data['position_x'])**2 +
            (data[pred_columns[1]] - data['position_y'])**2 +
            (data[pred_columns[2]] - data['position_z'])**2
        )
        data_copy = data.copy()
        data_copy['distance'] = distances
        logger.info(f"Calculated vertex distances for {len(data_copy)} events")
        return data_copy
        
    except Exception as e:
        logger.error(f"Error calculating vertex distances: {e}")
        raise


def apply_event_filters(data: pd.DataFrame, filter_tracks: bool = False, filter_cascades: bool = False) -> pd.DataFrame:
    """
    Apply event selection filters to the dataset.
    
    Args:
        data: Input DataFrame
        filter_tracks: If True, filter for muon neutrino charged current events
                      (pid=±14, interaction_type=1)
        filter_cascades: If True, filter for cascade events (everything except
                        muon neutrino charged current events)
        
    Returns:
        Filtered DataFrame
        
    Note:
        filter_tracks and filter_cascades are mutually exclusive. If both are True,
        filter_tracks takes precedence.
    """
    if not filter_tracks and not filter_cascades:
        return data
    
    initial_count = len(data)
    
    if filter_tracks:
        # Filter for muon neutrino charged current interactions
        # pid = ±14 (muon neutrinos), interaction_type = 1 (charged current)
        filtered_data = data[
            (np.abs(data['pid']) == 14) & 
            (data['interaction_type'] == 1)
        ].copy()
        filter_type = "track filters"
        
    else:  # filter_cascades must be True due to early return above
        # Filter for cascade events (everything except muon neutrino charged current)
        # This includes: electron neutrinos, tau neutrinos, and muon neutrino neutral current
        filtered_data = data[
            ~((np.abs(data['pid']) == 14) & (data['interaction_type'] == 1))
        ].copy()
        filter_type = "cascade filters"
    
    final_count = len(filtered_data)
    
    logger.info(f"Applied {filter_type}: {initial_count} → {final_count} events "
                f"({100 * final_count / initial_count:.1f}% retained)")
    
    return filtered_data


def compute_energy_binned_statistics(
    data: pd.DataFrame, 
    num_bins: int = DEFAULT_ENERGY_BINS,
    energy_range: Tuple[float, float] = DEFAULT_ENERGY_RANGE,
    min_events: int = DEFAULT_MIN_EVENTS_PER_BIN
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute distance statistics binned by neutrino energy.
    
    Args:
        data: DataFrame with 'distance' and 'energy' columns
        num_bins: Number of energy bins to create
        energy_range: Tuple of (min, max) log10(energy) values
        min_events: Minimum number of events required per bin
        
    Returns:
        Tuple of (bin_centers, statistics_dataframe)
        Statistics include median, 16th/84th percentiles (68% confidence), and count
    """
    # Create log energy column for binning
    data_copy = data.copy()
    data_copy['log_energy'] = np.log10(data_copy['energy'])
    
    # Create energy bins
    energy_bins = np.linspace(energy_range[0], energy_range[1], num=num_bins + 1)
    
    # Group data by energy bins and compute statistics
    grouped = data_copy.groupby(pd.cut(data_copy['log_energy'], bins=energy_bins))
    
    statistics = grouped['distance'].agg([
        'median',
        'count',
        lambda x: np.percentile(x, 16),  # 68% confidence lower bound
        lambda x: np.percentile(x, 84)   # 68% confidence upper bound
    ])
    
    statistics.columns = ['Median', 'Count', '68% Lower', '68% Upper']
    
    # Filter bins with insufficient statistics
    valid_bins = statistics['Count'] >= min_events
    statistics_filtered = statistics[valid_bins]
    
    # Calculate bin centers for plotting
    bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    bin_centers_filtered = bin_centers[valid_bins]
    
    logger.info(f"Computed statistics for {len(statistics_filtered)} energy bins "
                f"(filtered from {len(statistics)} total bins)")
    
    return bin_centers_filtered, statistics_filtered


def process_csv_file(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single CSV file according to the provided configuration.
    
    Args:
        config: Dictionary containing processing parameters:
            - csv_filepath: Path to the CSV file
            - pred_columns: List of predicted position column names
            - label: Label for plotting
            - filter_tracks: Whether to apply track filtering
            - filter_cascades: Whether to apply cascade filtering
            - show_in_ratio_plot: Whether to include in ratio plots
            
    Returns:
        Dictionary containing processed statistics and metadata
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        Exception: If processing fails
    """
    csv_filepath = Path(config['csv_filepath'])
    if not csv_filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_filepath}")
    
    try:
        # Load data
        logger.info(f"Processing file: {csv_filepath}")
        data = pd.read_csv(csv_filepath)
        logger.info(f"Loaded {len(data)} events from {csv_filepath.name}")
        
        # Apply filters
        if config.get('filter_tracks', False) or config.get('filter_cascades', False):
            data = apply_event_filters(
                data, 
                filter_tracks=config.get('filter_tracks', False),
                filter_cascades=config.get('filter_cascades', False)
            )
        
        # Calculate vertex distances
        data = calculate_vertex_distance(data, config['pred_columns'])
        
        # Compute energy-binned statistics
        bin_centers, statistics = compute_energy_binned_statistics(data)
        
        return {
            'label': config['label'],
            'bin_centers': bin_centers,
            'statistics': statistics,
            'show_in_ratio_plot': config.get('show_in_ratio_plot', True),
            'total_events': len(data)
        }
        
    except Exception as e:
        logger.error(f"Error processing {csv_filepath}: {e}")
        raise


def process_multiple_csv_files(csv_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process multiple CSV files according to their configurations.
    
    Args:
        csv_configs: List of configuration dictionaries for each CSV file
        
    Returns:
        List of processed statistics dictionaries
    """
    if not csv_configs:
        logger.warning("No CSV configurations provided")
        return []
    
    processed_data = []
    
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
# NUMPY/DATABASE DATA PROCESSING (ALTERNATIVE DATA SOURCE)
# =============================================================================

def compute_vertex_error_from_arrays(
    npy_data: np.ndarray, 
    db_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute vertex reconstruction errors from numpy arrays and database data.
    
    This function is used for processing data stored in numpy format with
    corresponding truth information in a database.
    
    Args:
        npy_data: Numpy array containing reconstruction results
        db_data: DataFrame containing truth information
        
    Returns:
        Merged DataFrame with computed vertex errors
    """
    # Convert numpy data to DataFrame
    df_reconstructed = pd.DataFrame(npy_data)
    
    # Filter for charged current events only
    df_reconstructed = df_reconstructed[df_reconstructed['iscc'] == True]
    
    # Merge with truth data
    df_merged = df_reconstructed.merge(db_data, on='event_no', how='inner')
    
    # Calculate cylindrical coordinate errors
    df_merged['rho_true'] = np.sqrt(
        df_merged['position_x']**2 + df_merged['position_y']**2
    )
    df_merged['rho_error'] = np.abs(df_merged['rho'] - df_merged['rho_true'])
    df_merged['z_error'] = np.abs(df_merged['z'] - df_merged['position_z'])
    
    # Calculate total 3D vertex error
    df_merged['vertex_error'] = np.sqrt(
        df_merged['rho_error']**2 + df_merged['z_error']**2
    )
    
    # Add log energy for binning
    df_merged['log_trueE'] = np.log10(df_merged['trueE'])
    
    logger.info(f"Computed vertex errors for {len(df_merged)} merged events")
    return df_merged


def bin_and_compute_statistics_from_arrays(
    df: pd.DataFrame, 
    num_bins: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute binned statistics for numpy/database processed data.
    
    Args:
        df: DataFrame with vertex errors and log energy
        num_bins: Number of energy bins
        
    Returns:
        Dictionary containing bin centers and statistics arrays
    """
    energy_min, energy_max = float(df['log_trueE'].min()), float(df['log_trueE'].max())
    bins = np.linspace(energy_min, energy_max, num_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    
    result = {
        'bin_centers': centers,
        'median': [],
        '68_lower': [],
        '68_upper': []
    }
    
    for i in range(len(bins) - 1):
        mask = (df['log_trueE'] >= bins[i]) & (df['log_trueE'] < bins[i+1])
        subset = df[mask]['vertex_error']
        
        if len(subset) > 0:
            result['median'].append(np.median(subset))
            result['68_lower'].append(np.percentile(subset, 16))
            result['68_upper'].append(np.percentile(subset, 84))
        else:
            result['median'].append(np.nan)
            result['68_lower'].append(np.nan)
            result['68_upper'].append(np.nan)
    
    return result


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def create_combined_plot(
    processed_data: List[Dict[str, Any]], 
    title: str,
    reference_label: str = "Reference",
    figure_size: Tuple[float, float] = (12, 10),
    dpi: int = 600,
    save_png: Optional[str] = None,
    save_pdf: Optional[str] = None,
    include_ratio_plot: bool = True
) -> None:
    """
    Create a combined plot showing vertex distance statistics and ratios.
    
    Args:
        processed_data: List of processed data dictionaries
        title: Plot title
        reference_label: Label of the reference dataset for ratio calculations
        figure_size: Figure size as (width, height) in inches
        dpi: Plot resolution in dots per inch
        save_png: Optional filepath to save plot as PNG (e.g., "vertex_plot.png")
        save_pdf: Optional filepath to save plot as PDF (e.g., "vertex_plot.pdf")
        include_ratio_plot: Whether to include the ratio comparison subplot (default: True)
    """
    if not processed_data:
        logger.warning("No data provided for plotting")
        return
    
    # Create subplots with conditional layout
    ax_ratio: Optional[Axes] = None
    if include_ratio_plot:
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, 
            figsize=figure_size, 
            dpi=dpi,
            gridspec_kw={'height_ratios': [3, 1]}, 
            sharex=True
        )
    else:
        fig, ax_main = plt.subplots(
            1, 1, 
            figsize=figure_size, 
            dpi=dpi
        )
    
    reference_median = None
    
    # Plot main data with error bands
    for i, data in enumerate(processed_data):
        color = COLORBLIND_FRIENDLY_COLORS[i % len(COLORBLIND_FRIENDLY_COLORS)]
        
        x = data['bin_centers']
        y = data['statistics']['Median']
        y_lower = data['statistics']['68% Lower']
        y_upper = data['statistics']['68% Upper']
        
        # Main plot
        ax_main.plot(x, y, marker='o', linestyle='-', 
                    label=data['label'], color=color, linewidth=2)
        ax_main.fill_between(x, y_lower, y_upper, color=color, alpha=0.2)
        
        # Store reference data for ratio calculation
        if data['label'] == reference_label:
            reference_median = np.array(y)
            logger.info(f"Using '{reference_label}' as reference for ratio plots")
    
    # Plot ratios relative to reference (only if ratio plot is included)
    if include_ratio_plot and reference_median is not None and ax_ratio is not None:
        for i, data in enumerate(processed_data):
            if data['show_in_ratio_plot'] and data['label'] != reference_label:
                color = COLORBLIND_FRIENDLY_COLORS[i % len(COLORBLIND_FRIENDLY_COLORS)]
                y = np.array(data['statistics']['Median'])
                x = data['bin_centers']
                
                # Avoid division by zero
                valid_ref = reference_median > 0
                if np.any(valid_ref):
                    ratio = np.divide(y, reference_median, 
                                    out=np.full_like(y, np.nan), 
                                    where=valid_ref)
                    ax_ratio.plot(x[valid_ref], ratio[valid_ref], 
                                marker='o', linestyle='-', color=color, linewidth=2)
    
    # Configure main plot
    ax_main.set_yticks([10, 20, 40, 60, 80])
    ax_main.set_xlim(DEFAULT_ENERGY_RANGE)
    ax_main.set_ylim(0, 105)
    ax_main.set_ylabel("Vertex Distance (m)")
    ax_main.set_title(title)
    
    # Adjust legend positioning based on whether ratio plot is included
    if include_ratio_plot:
        ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08))
    else:
        ax_main.legend(loc='best')
        ax_main.set_xlabel(r'$\log_{10}(\mathrm{True~Neutrino~Energy~/~GeV})$')
    
    ax_main.grid(True, alpha=0.3)
    
    # Configure ratio plot (only if included)
    if include_ratio_plot:
        ax_ratio.set_ylim(0.9, 1.2)
        ax_ratio.set_yticks([0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
        ax_ratio.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        ax_ratio.set_ylabel('Ratio over Reference')
        ax_ratio.set_xlabel(r'$\log_{10}(\mathrm{True~Neutrino~Energy~/~GeV})$')
        ax_ratio.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots if requested
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

def create_example_config() -> List[Dict[str, Any]]:
    """
    Create example configuration for CSV processing.
    
    Returns:
        List of example configuration dictionaries
    """
    return [
        {
            "csv_filepath": "path/to/your/file1.csv",
            "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
            "label": "Model 1 (Tracks)",
            "filter_tracks": True,
            "filter_cascades": False,
            "show_in_ratio_plot": True
        },
        {
            "csv_filepath": "path/to/your/file2.csv",
            "pred_columns": ['vertex_x_pred', 'vertex_y_pred', 'vertex_z_pred'],
            "label": "Model 2 (Cascades)",
            "filter_tracks": False,
            "filter_cascades": True,
            "show_in_ratio_plot": True
        },
        {
            "csv_filepath": "path/to/your/reference.csv",
            "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
            "label": "Reference (All Events)",
            "filter_tracks": False,
            "filter_cascades": False,
            "show_in_ratio_plot": False
        }
    ]


def main():
    """
    Main execution function for vertex analysis plotting.
    
    Configure your CSV files in the csv_configs list below, then run this script.
    Each configuration should specify:
    - csv_filepath: Path to your CSV file
    - pred_columns: List of [x, y, z] column names for predicted positions
    - label: Display name for the dataset
    - filter_tracks: Whether to filter for muon neutrino CC events (tracks)
    - filter_cascades: Whether to filter for cascade events (everything except muon neutrino CC)
    - show_in_ratio_plot: Whether to include in ratio comparison
    
    Note: filter_tracks and filter_cascades are mutually exclusive.
    """
    
    # =============================================================================
    # CONFIGURATION SECTION - MODIFY THIS FOR YOUR DATA
    # =============================================================================
    
    # Define the four analysis configurations
    analysis_configs = [
        {
            "name": "numu_tracks",
            "title": "Vertex Reconstruction Accuracy Comparison (νμ Tracks Only)",
            "reference_label": r"TANGO $\alpha=0.04$ (Reference)",
            "filename": "vertex_analysis_numu_tracks.png",
            "csv_configs": [
                # {
                #     "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.026_example/results.csv",
                #     "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                #     "label": r"IceMix Tiny $\alpha=0.026$",
                #     "filter_tracks": True,
                #     "filter_cascades": False,
                #     "show_in_ratio_plot": True
                # },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.040_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.040$",
                    "filter_tracks": True,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True
                },
                                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix $\alpha=0.040$",
                    "filter_tracks": True,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True
                },
                # {
                #     "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.060_example/results.csv",
                #     "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                #     "label": r"IceMix Tiny $\alpha=0.060$",
                #     "filter_tracks": True,
                #     "filter_cascades": False,
                #     "show_in_ratio_plot": True
                # },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"TANGO $\alpha=0.04$ (Reference)",
                    "filter_tracks": True,
                    "filter_cascades": False,
                    "show_in_ratio_plot": False
                }
            ]
        },
        {
            "name": "numu_cascades",
            "title": "Vertex Reconstruction Accuracy Comparison (νμ Cascades Only)",
            "reference_label": r"TANGO $\alpha=0.04$ (Reference)",
            "filename": "vertex_analysis_numu_cascades.png",
            "csv_configs": [
                # {
                #     "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.026_example/results.csv",
                #     "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                #     "label": r"IceMix Tiny $\alpha=0.026$",
                #     "filter_tracks": False,
                #     "filter_cascades": True,
                #     "show_in_ratio_plot": True
                # },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.040_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.040$",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True
                },
                                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix $\alpha=0.040$",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True
                },
                # {
                #     "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.060_example/results.csv",
                #     "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                #     "label": r"IceMix Tiny $\alpha=0.060$",
                #     "filter_tracks": False,
                #     "filter_cascades": True,
                #     "show_in_ratio_plot": True
                # },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"TANGO $\alpha=0.04$ (Reference)",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": False
                }
            ]
        },
        {
            "name": "nue_cascades",
            "title": "Vertex Reconstruction Accuracy Comparison (νe Cascades Only)",
            "reference_label": r"TANGO $\alpha=0.04$ (Reference)",
            "filename": "vertex_analysis_nue_cascades.png",
            "csv_configs": [
                # {
                #     "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.026_example/results.csv",
                #     "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                #     "label": r"IceMix Tiny $\alpha=0.026$",
                #     "filter_tracks": False,
                #     "filter_cascades": True,
                #     "show_in_ratio_plot": True
                # },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.040_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.040$",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True
                },
                                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix $\alpha=0.040$",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": True
                },
                # {
                #     "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.060_example/results.csv",
                #     "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                #     "label": r"IceMix Tiny $\alpha=0.060$",
                #     "filter_tracks": False,
                #     "filter_cascades": True,
                #     "show_in_ratio_plot": True
                # },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"TANGO $\alpha=0.04$ (Reference)",
                    "filter_tracks": False,
                    "filter_cascades": True,
                    "show_in_ratio_plot": False
                }
            ]
        },
        {
            "name": "all_events",
            "title": "Vertex Reconstruction Accuracy Comparison (All Events)",
            "reference_label": r"TANGO $\alpha=0.04$ (Reference)",
            "filename": "vertex_analysis_all_events.png",
            "csv_configs": [
                # {
                #     "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.026_example/results.csv",
                #     "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                #     "label": r"IceMix Tiny $\alpha=0.026$",
                #     "filter_tracks": False,
                #     "filter_cascades": False,
                #     "show_in_ratio_plot": True
                # },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.040_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix Tiny $\alpha=0.040$",
                    "filter_tracks": False,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True
                },
                                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_example/results.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"IceMix $\alpha=0.040$",
                    "filter_tracks": False,
                    "filter_cascades": False,
                    "show_in_ratio_plot": True
                },
                # {
                #     "csv_filepath": "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_alpha_0.060_example/results.csv",
                #     "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                #     "label": r"IceMix Tiny $\alpha=0.060$",
                #     "filter_tracks": False,
                #     "filter_cascades": False,
                #     "show_in_ratio_plot": True
                # },
                {
                    "csv_filepath": "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/baseline/JointLargeTC0.04results_LRNEW.csv",
                    "pred_columns": ['pos_x_pred', 'pos_y_pred', 'pos_z_pred'],
                    "label": r"TANGO $\alpha=0.04$ (Reference)",
                    "filter_tracks": False,
                    "filter_cascades": False,
                    "show_in_ratio_plot": False
                }
            ]
        }
    ]
    
    # Save options (set to None to disable saving)
    # Examples:
    # save_png_path = "vertex_analysis_plot.png"  # Save as PNG
    # save_png_path = "plots/my_analysis.png"     # Save in subdirectory
    # save_png_path = None                        # Don't save PNG
    base_path = 'carlos_tests/icemix_tiny/plots'
    save_png_path = "vertex_analysis_plot.png"  # Set to None to disable PNG saving
    save_pdf_path = "vertex_analysis_plot.pdf"  # Set to None to disable PDF saving
    # Join the base path and the save path
    save_png_path = os.path.join(base_path, save_png_path) if save_png_path else None
    save_pdf_path = os.path.join(base_path, save_pdf_path) if save_pdf_path else None
    
    # =============================================================================
    # EXECUTION
    # =============================================================================
    
    try:
        # Process each analysis configuration
        for config in analysis_configs:
            logger.info(f"Starting vertex analysis for: {config['name']}")
            
            # Process all CSV files for this configuration
            processed_statistics = process_multiple_csv_files(config['csv_configs'])
            
            if not processed_statistics:
                logger.error(f"No data was successfully processed for {config['name']}. Please check your configurations.")
                continue
            
            # Create visualization
            save_png_path = os.path.join(base_path, config['filename'])
            
            create_combined_plot(
                processed_statistics, 
                title=config['title'],
                reference_label=config['reference_label'],
                save_png=save_png_path,
                save_pdf=None,
                include_ratio_plot=True,
            )
            
            logger.info(f"Completed analysis for: {config['name']}")
        
        logger.info("All four vertex analysis variations completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
