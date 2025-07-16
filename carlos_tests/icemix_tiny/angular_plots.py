"""
Angular Resolution Analysis for Neutrino Direction Reconstruction

This module provides tools for analyzing the angular resolution of neutrino
direction reconstruction models by calculating angular differences between
true and predicted directions, and generating comparative plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
import os

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
FIGURE_SIZE = (14*1.5, 8*1.5)

# Analysis constants
ELECTRON_NEUTRINO_PID = 12
MUON_NEUTRINO_PID = 14
TAU_NEUTRINO_PID = 16
CHARGED_CURRENT_INTERACTION = 1
DEFAULT_ENERGY_BINS = 20
LOG_ENERGY_MIN = 1.0
LOG_ENERGY_MAX = 4.0
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

# Configure matplotlib globally
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
        "legend.fontsize": LABEL_FONT_SIZE
    })

# ============================================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================================

def calculate_angular_difference(true_azimuth: Union[np.ndarray, Any], true_zenith: Union[np.ndarray, Any],
                               pred_x: Union[np.ndarray, Any], pred_y: Union[np.ndarray, Any], 
                               pred_z: Union[np.ndarray, Any]) -> np.ndarray:
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
    # Convert inputs to numpy arrays for consistent handling
    true_azimuth = np.asarray(true_azimuth)
    true_zenith = np.asarray(true_zenith)
    pred_x = np.asarray(pred_x)
    pred_y = np.asarray(pred_y)
    pred_z = np.asarray(pred_z)
    
    # Convert Cartesian predicted coordinates back to spherical
    # pred_azimuth = arctan2(pred_y, pred_x)
    # pred_zenith = arccos(pred_z) (but we'll use atan2 for better numerical stability)
    pred_azimuth = np.arctan2(pred_y, pred_x)
    pred_zenith = np.arctan2(np.sqrt(pred_x**2 + pred_y**2), pred_z)
    
    # Use haversine formula for numerically stable angular separation
    # This is much more stable than arccos for small angles
    delta_azimuth = pred_azimuth - true_azimuth
    delta_zenith = pred_zenith - true_zenith
    
    # Haversine formula
    # hav(Δψ) = sin²(Δφ/2) + sin(φ1)sin(φ2)sin²(Δθ/2)
    # where φ is zenith (colatitude) and θ is azimuth
    haversine = (np.sin(delta_zenith / 2.0)**2 + 
                 np.sin(true_zenith) * np.sin(pred_zenith) * np.sin(delta_azimuth / 2.0)**2)
    
    # Guard against tiny negative values due to round-off
    haversine = np.clip(haversine, 0.0, 1.0)
    
    # Angular separation: Δψ = 2 * arcsin(sqrt(hav))
    angular_separation = 2.0 * np.arcsin(np.sqrt(haversine))
    
    # Return angular difference in degrees
    return np.degrees(angular_separation)


def load_and_filter_data(csv_filepath: str, 
                        include_charged_current: bool = True,
                        include_neutral_current: bool = True,
                        neutrino_pid: int = ELECTRON_NEUTRINO_PID) -> Any:
    """
    Load CSV data and apply filters for neutrino type and interaction type.
    
    Parameters:
    -----------
    csv_filepath : str
        Path to the CSV file containing neutrino data
    include_charged_current : bool
        Whether to include charged current interactions
    include_neutral_current : bool
        Whether to include neutral current interactions
    neutrino_pid : int
        Particle ID for the neutrino type to analyze
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe with columns: ['event_no', 'energy', 'angle']
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
            return pd.DataFrame(columns=["event_no", "energy", "angle"])
        else:
            logger.info("Keeping both charged and neutral current interactions")
        
        logger.info(f"After interaction type filter: {len(df)} events")
        
        # Calculate angular differences
        df["angle"] = calculate_angular_difference(
            df["azimuth"], df["zenith"],
            df["dir_x_pred"], df["dir_y_pred"], df["dir_z_pred"]
        )
        
        return df[["event_no", "energy", "angle"]]
        
    except FileNotFoundError:
        logger.error(f"File not found: {csv_filepath}")
        raise
    except KeyError as e:
        logger.error(f"Required column missing from CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise


def compute_angular_statistics(df: pd.DataFrame, 
                             energy_column: str = "energy",
                             num_bins: int = DEFAULT_ENERGY_BINS) -> Dict[str, List[float]]:
    """
    Compute angular resolution statistics binned by energy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing energy and angle columns
    energy_column : str
        Name of the energy column
    num_bins : int
        Number of energy bins to use
    
    Returns:
    --------
    Dict[str, List[float]]
        Dictionary containing bin centers, median values, and 68% confidence intervals
    """
    # Create log energy column
    df = df.copy()
    df["log_energy"] = np.log10(df[energy_column])
    
    # Define energy bins
    energy_bins = np.linspace(LOG_ENERGY_MIN, LOG_ENERGY_MAX, num_bins)
    bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])
    
    # Initialize statistics storage
    statistics = {
        "bin_centers": bin_centers.tolist(),
        "median": [],
        "percentile_16": [],  # Lower bound of 68% interval
        "percentile_84": []   # Upper bound of 68% interval
    }
    
    # Calculate statistics for each energy bin
    for i in range(len(energy_bins) - 1):
        bin_mask = (df["log_energy"] >= energy_bins[i]) & (df["log_energy"] < energy_bins[i + 1])
        bin_angles = df[bin_mask]["angle"]
        
        if len(bin_angles) > 0:
            statistics["median"].append(float(np.median(bin_angles)))
            statistics["percentile_16"].append(float(np.percentile(bin_angles, 16)))
            statistics["percentile_84"].append(float(np.percentile(bin_angles, 84)))
            logger.debug(f"Bin {i}: {len(bin_angles)} events, median = {statistics['median'][-1]:.2f}°")
        else:
            # Handle empty bins
            statistics["median"].append(np.nan)
            statistics["percentile_16"].append(np.nan)
            statistics["percentile_84"].append(np.nan)
            logger.warning(f"Bin {i} is empty")
    
    return statistics


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_comparison_plot(model_results: List[Tuple[Dict, str, bool]], 
                         reference_stats: Dict[str, List[float]],
                         title: str = r"GRECO $\nu_e$ Median Angular Resolution",
                         save_png: Optional[str] = None,
                         save_pdf: Optional[str] = None,
                         show_plot: bool = True,
                         include_ratio_plot: bool = True,
                         log_scale_y: bool = False,
                         show_theoretical_limit: bool = False) -> None:
    """
    Create a comparison plot showing angular resolution vs energy for multiple models.
    
    Parameters:
    -----------
    model_results : List[Tuple[Dict, str, bool]]
        List of tuples containing (statistics_dict, label, show_in_ratio_plot)
    reference_stats : Dict[str, List[float]]
        Reference statistics for ratio calculation
    title : str
        Plot title
    save_png : Optional[str]
        If provided, save plot as PNG to this filepath
    save_pdf : Optional[str]
        If provided, save plot as PDF to this filepath
    show_plot : bool
        Whether to display the plot interactively
    include_ratio_plot : bool
        Whether to include the ratio comparison subplot (default: True)
    log_scale_y : bool
        Whether to use logarithmic scale for the y-axis (default: False)
    show_theoretical_limit : bool
        Whether to show the theoretical limit line: 0.7 deg * (E/TeV)^-0.7 (default: False)
    """
    # Create figure with subplots
    if include_ratio_plot:
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, figsize=FIGURE_SIZE, dpi=FIGURE_DPI,
            gridspec_kw={'height_ratios': [3, 1]}, 
            sharex=True
        )
        # Plot ratio comparison
        _plot_ratio_comparison(ax_ratio, model_results, reference_stats, show_theoretical_limit)
    else:
        fig, ax_main = plt.subplots(
            1, 1, figsize=FIGURE_SIZE, dpi=FIGURE_DPI
        )
    
    # Plot main comparison
    _plot_main_comparison(ax_main, model_results, title, include_ratio_plot, log_scale_y, show_theoretical_limit)
    
    # Finalize plot
    plt.tight_layout()
    
    # Save plots if requested
    if save_png:
        logger.info(f"Saving plot as PNG: {save_png}")
        plt.savefig(save_png, format='png', dpi=FIGURE_DPI, bbox_inches='tight')
        
    if save_pdf:
        logger.info(f"Saving plot as PDF: {save_pdf}")
        plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close figure to free memory if not showing


def _plot_main_comparison(ax: plt.Axes, 
                         model_results: List[Tuple[Dict, str, bool]], 
                         title: str,
                         include_ratio_plot: bool = True,
                         log_scale_y: bool = False,
                         show_theoretical_limit: bool = False) -> None:
    """Plot the main angular resolution comparison."""
    for i, (stats, label, _) in enumerate(model_results):
        color = COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)]
        
        # Plot median line
        ax.plot(stats["bin_centers"], stats["median"],
                marker='o', linestyle='-', color=color, 
                label=label, linewidth=2, markersize=6)
        
        # Plot confidence interval
        ax.fill_between(stats["bin_centers"],
                       stats["percentile_16"], stats["percentile_84"],
                       color=color, alpha=0.2)
    
    # Plot theoretical limit if requested
    if show_theoretical_limit:
        # Theoretical limit: 0.7 deg * (E/TeV)^-0.7
        # Create a fine energy range for smooth curve
        energy_range = np.linspace(LOG_ENERGY_MIN, LOG_ENERGY_MAX, 100)
        # Convert from log10(GeV) to TeV: E_TeV = 10^(log10_GeV) / 1000
        energy_tev = (10**energy_range) / 1000.0
        # Calculate theoretical limit: 0.7 * (E/TeV)^-0.7
        theoretical_limit = 0.7 * (energy_tev**(-0.7))
        
        ax.plot(energy_range, theoretical_limit, 
               color='red', linestyle='--', linewidth=2,
               label=r'Theoretical Limit: 0.7° × (E/TeV)$^{-0.7}$', 
               alpha=0.8)
    
    # Configure main plot
    ax.set_xlim(LOG_ENERGY_MIN, LOG_ENERGY_MAX)
    
    # Set y-axis scale and limits
    if log_scale_y:
        ax.set_yscale('log')
        # For log scale, we need to set a sensible lower limit (can't be 0)
        # Find the minimum non-zero value from all percentile_16 data
        min_values = []
        for stats, _, _ in model_results:
            min_val = np.min([v for v in stats["percentile_16"] if not np.isnan(v) and v > 0])
            if min_val > 0:
                min_values.append(min_val)
        
        if min_values:
            y_min = min(min_values) * 0.8  # Add some padding
            ax.set_ylim(y_min, ANGLE_PLOT_MAX)
        else:
            ax.set_ylim(0.1, ANGLE_PLOT_MAX)  # Fallback values
            
        # Enable minor ticks only on y-axis and add fine minor gridlines for log scale
        from matplotlib.ticker import LogLocator, FixedFormatter
        
        # Set up major tick locations  
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        
        # Create custom minor ticks with explicit positions and labels
        # Generate minor tick positions (2,3,4,5,6,7,8,9 for each decade)
        minor_ticks = []
        minor_labels = []
        
        # Get the current y-limits to determine the range
        y_min, y_max = ax.get_ylim()
        
        # Generate minor ticks for each decade
        for decade in [0.1, 1, 10, 100]:
            if decade >= y_min and decade <= y_max:
                for sub in [2, 3, 4, 5, 6, 7, 8, 9]:
                    tick_val = decade * sub
                    if tick_val >= y_min and tick_val <= y_max:
                        minor_ticks.append(tick_val)
                        minor_labels.append(f'{tick_val:g}')  # Format without unnecessary decimals
        
        # Set minor ticks with explicit positions and labels
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_yticklabels(minor_labels, minor=True, fontsize=TICK_FONT_SIZE-2)
        
        # Configure tick appearance
        ax.tick_params(axis='y', which='minor', length=3)  # Show y minor ticks
        ax.tick_params(axis='y', which='major', labelsize=TICK_FONT_SIZE)  # Major tick size
        ax.tick_params(axis='x', which='minor', length=0)  # Hide x minor ticks
        
        ax.grid(True, which='major', alpha=0.3)  # Major gridlines
        ax.grid(True, which='minor', alpha=0.15, linewidth=0.5, axis='y')  # Fine minor gridlines only on y-axis
    else:
        ax.set_ylim(0, ANGLE_PLOT_MAX)
        ax.grid(True, alpha=0.3)  # Regular gridlines for linear scale
    
    ax.set_ylabel("Angular Difference (°)")
    ax.set_title(title)
    
    # Adjust legend positioning based on whether ratio plot is included
    if include_ratio_plot:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    else:
        ax.legend(loc="best")
        ax.set_xlabel(r"$\log_{10}(\mathrm{True~Neutrino~Energy~/~GeV})$")


def _plot_ratio_comparison(ax: plt.Axes, 
                          model_results: List[Tuple[Dict, str, bool]], 
                          reference_stats: Dict[str, List[float]],
                          show_theoretical_limit: bool = False) -> None:
    """Plot the ratio comparison in the lower panel."""
    reference_median = np.array(reference_stats["median"])
    
    for i, (stats, label, show_ratio) in enumerate(model_results):
        if show_ratio:
            color = COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)]
            
            # Calculate ratio with handling for NaN values
            model_median = np.array(stats["median"])
            ratio = model_median / reference_median
            
            ax.plot(stats["bin_centers"], ratio,
                   marker='o', linestyle='-', color=color, 
                   label=label, linewidth=2, markersize=6)
    
    # Configure ratio plot
    ax.axhline(1, color='gray', linestyle='--', linewidth=1)
    ytick_values = [0.9, 0.93, 0.96, 1.0, 1.03, 1.06, 1.09]
    ax.set_yticks(ytick_values)
    ax.set_ylabel("Ratio over Reference")
    ax.set_xlabel(r"$\log_{10}(\mathrm{True~Neutrino~Energy~/~GeV})$")
    ax.grid(True, alpha=0.3)

    if show_theoretical_limit:
        # Calculate theoretical limit ratio compared to reference
        # Theoretical limit: 0.7 deg * (E/TeV)^-0.7
        energy_range = np.linspace(LOG_ENERGY_MIN, LOG_ENERGY_MAX, 100)
        # Convert from log10(GeV) to TeV: E_TeV = 10^(log10_GeV) / 1000
        energy_tev = (10**energy_range) / 1000.0
        # Calculate theoretical limit: 0.7 * (E/TeV)^-0.7
        theoretical_limit = 0.7 * (energy_tev**(-0.7))
        
        # Calculate ratio of theoretical limit to reference
        reference_interp = np.interp(energy_range, reference_stats["bin_centers"], reference_stats["median"])
        theoretical_ratio = theoretical_limit / reference_interp
        
        ax.plot(energy_range, theoretical_ratio, 
               color='red', linestyle='--', linewidth=2,
               label='Theoretical Limit Ratio', alpha=0.8)
        ax.legend()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution function for angular resolution analysis.
    
    Configure the CSV files and analysis parameters below, then run the script
    to generate comparison plots.
    """
    # Setup plotting style
    setup_matplotlib_style()
    
    # ========================================================================
    # CONFIGURATION SECTION - MODIFY THESE PARAMETERS
    # ========================================================================
    
    # List of CSV files to analyze: (filepath, label, show_in_ratio_plot)
    csv_files = [
        (
            "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/my_numu_database_part_1 (1)/dynedgeTITO_direction_example/results.csv",
            r"IceMix Tiny $\alpha=0.01$ w/ Dropout (Cascades)",
            True
        ),
        (
            "carlos_tests/icemix_tiny/baseline/JointLargeTC0.01results_LRNEW.csv",
            r"TANGO $\alpha=0.01$ (Cascades)",
            False
        )
        # Example entries (uncomment and modify as needed):
        # ("path/to/model1.csv", "Model 1", True),
        # ("path/to/model2.csv", "Model 2", True),
        # ("path/to/model3.csv", "Model 3", False),
    ]
    
    # Reference CSV file for ratio calculations
    reference_csv_path = "carlos_tests/icemix_tiny/baseline/JointLargeTC0.01results_LRNEW.csv"
    # "path/to/reference.csv"
    
    # Analysis parameters
    include_charged_current = False
    include_neutral_current = True
    neutrino_type = MUON_NEUTRINO_PID  # 12 for electron neutrino, 14 for muon, 16 for tau
    
    # Plot saving options
    base_path = 'carlos_tests/icemix_tiny/plots'
    save_png_filename = 'angular_resolution_comparison.png'  # e.g., "angular_resolution_comparison.png"
    save_pdf_filename = None  # e.g., "angular_resolution_comparison.pdf"
    
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Join the base path and the save path
    save_png_path = os.path.join(base_path, save_png_filename) if save_png_filename else None
    save_pdf_path = os.path.join(base_path, save_pdf_filename) if save_pdf_filename else None
    show_interactive_plot = False  # Set to False if you only want to save without displaying
    
    # Plot configuration options
    include_ratio_plot = True  # Set to False to only show the main comparison plot without ratio
    log_scale_y = True  # Set to True to use logarithmic scale for the y-axis
    show_theoretical_limit = False # Set to True to show the theoretical limit
    
    # ========================================================================
    # VALIDATION AND EXECUTION
    # ========================================================================
    
    if not csv_files:
        logger.warning("No CSV files configured. Please add file paths to csv_files list.")
        logger.info("Example configuration:")
        logger.info('csv_files = [("model1.csv", "Model 1", True)]')
        return
    
    if reference_csv_path is None:
        logger.warning("No reference CSV configured. Please set reference_csv_path.")
        return
    
    try:
        # Process model files
        logger.info(f"Processing {len(csv_files)} model files...")
        model_results = []
        
        for filepath, label, show_ratio in csv_files:
            logger.info(f"Processing {label}: {filepath}")
            df = load_and_filter_data(
                filepath, 
                include_charged_current, 
                include_neutral_current,
                neutrino_type
            )
            
            if len(df) == 0:
                logger.warning(f"No data found for {label} after filtering")
                continue
                
            stats = compute_angular_statistics(df)
            model_results.append((stats, label, show_ratio))
        
        # Process reference file
        logger.info(f"Processing reference file: {reference_csv_path}")
        reference_df = load_and_filter_data(
            reference_csv_path, 
            include_charged_current, 
            include_neutral_current,
            neutrino_type
        )
        
        if len(reference_df) == 0:
            logger.error("No data found in reference file after filtering")
            return
            
        reference_stats = compute_angular_statistics(reference_df)
        
        # Generate comparison plot
        logger.info("Generating comparison plot...")
        neutrino_name = {12: r"$\nu_e$", 14: r"$\nu_\mu$", 16: r"$\nu_\tau$"}.get(
            neutrino_type, f"PID {neutrino_type}"
        )
        plot_title = f"GRECO {neutrino_name} Median Angular Resolution"
        
        create_comparison_plot(
            model_results, 
            reference_stats, 
            plot_title,
            save_png=save_png_path,
            save_pdf=save_pdf_path,
            show_plot=show_interactive_plot,
            include_ratio_plot=include_ratio_plot,
            log_scale_y=log_scale_y,
            show_theoretical_limit=show_theoretical_limit
        )
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
