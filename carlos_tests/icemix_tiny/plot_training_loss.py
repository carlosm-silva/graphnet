#!/usr/bin/env python3
"""
Training Loss Plotting Script

This script combines training metrics from multiple PyTorch Lightning versions
and creates comprehensive loss vs epoch plots showing both training and validation losses.

Author: Carlos Filho
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

# Font configuration for publication-quality plots
LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 20
TITLE_FONT_SIZE = 24
LEGEND_FONT_SIZE = 18
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
    "legend.fontsize": LEGEND_FONT_SIZE,
})

# Colors for training and validation
TRAIN_COLOR = "#E69F00"  # Orange
VAL_COLOR = "#56B4E9"    # Blue
MIN_VAL_COLOR = "#D55E00"  # Red for minimum validation loss point


def parse_metrics_csv(filepath: str) -> pd.DataFrame:
    """
    Parse PyTorch Lightning metrics CSV file.
    
    Args:
        filepath: Path to the metrics.csv file
        
    Returns:
        DataFrame with columns: epoch, train_loss, val_loss
    """
    logger.info(f"Parsing metrics from: {filepath}")
    
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Separate training and validation loss entries
    train_entries = df[(df['train_loss'].notna()) & (df['val_loss'].isna())][['epoch', 'train_loss']].copy()
    val_entries = df[(df['val_loss'].notna()) & (df['train_loss'].isna())][['epoch', 'val_loss']].copy()
    
    # Group by epoch to get one value per epoch
    train_by_epoch = train_entries.groupby('epoch')['train_loss'].last().reset_index()
    val_by_epoch = val_entries.groupby('epoch')['val_loss'].last().reset_index()
    
    # Merge training and validation data
    metrics = pd.merge(train_by_epoch, val_by_epoch, on='epoch', how='outer').sort_values('epoch')
    
    logger.info(f"Found {len(metrics)} epochs with epoch range: {metrics['epoch'].min()}-{metrics['epoch'].max()}")
    logger.info(f"Training loss entries: {train_by_epoch['train_loss'].notna().sum()}")
    logger.info(f"Validation loss entries: {val_by_epoch['val_loss'].notna().sum()}")
    
    return metrics


def combine_metrics_from_versions(version_paths: List[str]) -> pd.DataFrame:
    """
    Combine metrics from multiple training versions.
    
    Args:
        version_paths: List of paths to metrics.csv files
        
    Returns:
        Combined DataFrame with all epochs
    """
    logger.info("Combining metrics from multiple versions...")
    
    all_metrics = []
    
    for path in version_paths:
        if Path(path).exists():
            metrics = parse_metrics_csv(path)
            all_metrics.append(metrics)
        else:
            logger.warning(f"File not found: {path}")
    
    if not all_metrics:
        raise FileNotFoundError("No valid metrics files found!")
    
    # Combine all metrics
    combined = pd.concat(all_metrics, ignore_index=True)
    
    # Remove duplicates, keeping the last occurrence of each epoch
    combined = combined.drop_duplicates(subset=['epoch'], keep='last').sort_values('epoch').reset_index(drop=True)
    
    logger.info(f"Combined metrics: {len(combined)} total epochs ({combined['epoch'].min()}-{combined['epoch'].max()})")
    
    return combined


def create_loss_plot(metrics: pd.DataFrame, save_path: Optional[str] = None, show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Create a comprehensive loss vs epoch plot.
    
    Args:
        metrics: DataFrame with epoch, train_loss, val_loss columns
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Tuple of (figure, axes) objects
    """
    logger.info("Creating loss vs epoch plot...")
    
    # Create figure and axis
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    # Plot training loss
    train_mask = metrics['train_loss'].notna()
    if train_mask.any():
        ax.plot(metrics.loc[train_mask, 'epoch'], 
                metrics.loc[train_mask, 'train_loss'], 
                color=TRAIN_COLOR, 
                linewidth=2.5, 
                marker='o', 
                markersize=6,
                label='Training Loss',
                alpha=0.9)
    
    # Plot validation loss
    val_mask = metrics['val_loss'].notna()
    if val_mask.any():
        val_data = metrics.loc[val_mask]
        ax.plot(val_data['epoch'], 
                val_data['val_loss'], 
                color=VAL_COLOR, 
                linewidth=2.5, 
                marker='s', 
                markersize=6,
                label='Validation Loss',
                alpha=0.9)
        
        # Highlight the minimum validation loss point
        min_val_idx = val_data['val_loss'].idxmin()
        min_val_epoch = val_data.loc[min_val_idx, 'epoch']
        min_val_loss = val_data.loc[min_val_idx, 'val_loss']
        ax.plot(min_val_epoch, min_val_loss, 
                color=MIN_VAL_COLOR, 
                marker='*', 
                markersize=15,
                label=f'Min Val Loss (Epoch {min_val_epoch})',
                markeredgecolor='black',
                markeredgewidth=1,
                zorder=10)
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Loss (log scale)', fontsize=LABEL_FONT_SIZE)
    ax.set_title('Training and Validation Loss vs Epoch', fontsize=TITLE_FONT_SIZE, pad=20)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', which='major')
    ax.grid(True, alpha=0.15, linestyle=':', which='minor')
    ax.minorticks_on()
    
    # Add vertical lines at learning rate reduction epochs (6n-1)
    max_epoch = metrics['epoch'].max() if len(metrics) > 0 else 60
    lr_epochs = [6*n - 1 for n in range(1, int((max_epoch + 6) // 6) + 1) if 6*n - 1 <= max_epoch]
    
    for i, epoch in enumerate(lr_epochs):
        # Add vertical line
        ax.axvline(x=epoch, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Add label only for the first occurrence to avoid cluttering
        if i == 0:
            ax.text(epoch + 0.5, ax.get_ylim()[1] * 0.8, 'Learning Rate\nHalved', 
                   rotation=0, fontsize=12, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Set axis limits with some padding
    if len(metrics) > 0:
        ax.set_xlim(metrics['epoch'].min() - 0.5, metrics['epoch'].max() + 0.5)
        
        # Calculate y-limits considering both losses
        all_losses = pd.concat([metrics['train_loss'].dropna(), metrics['val_loss'].dropna()])
        if len(all_losses) > 0:
            y_min, y_max = all_losses.min(), all_losses.max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig, ax


def print_training_summary(metrics: pd.DataFrame) -> None:
    """Print a summary of the training progress."""
    logger.info("Training Summary:")
    logger.info("=" * 50)
    
    train_data = metrics['train_loss'].dropna()
    val_data = metrics['val_loss'].dropna()
    
    if len(train_data) > 0:
        logger.info(f"Training Loss:")
        logger.info(f"  Initial: {train_data.iloc[0]:.6f}")
        logger.info(f"  Final:   {train_data.iloc[-1]:.6f}")
        logger.info(f"  Min:     {train_data.min():.6f} (epoch {metrics.loc[train_data.idxmin(), 'epoch']})")
        logger.info(f"  Max:     {train_data.max():.6f} (epoch {metrics.loc[train_data.idxmax(), 'epoch']})")
    
    if len(val_data) > 0:
        logger.info(f"Validation Loss:")
        logger.info(f"  Initial: {val_data.iloc[0]:.6f}")
        logger.info(f"  Final:   {val_data.iloc[-1]:.6f}")
        logger.info(f"  Min:     {val_data.min():.6f} (epoch {metrics.loc[val_data.idxmin(), 'epoch']})")
        logger.info(f"  Max:     {val_data.max():.6f} (epoch {metrics.loc[val_data.idxmax(), 'epoch']})")
    
    logger.info("=" * 50)


def main():
    """Main function to create training loss plots."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Define paths to metrics files relative to script location
    base_path = script_dir / "logs" / "training_logs"
    version_paths = [
        base_path / "version_5" / "metrics.csv",  # epochs 0-29
        base_path / "version_8" / "metrics.csv",  # epochs 30-59
    ]
    
    try:
        # Combine metrics from both versions
        combined_metrics = combine_metrics_from_versions([str(p) for p in version_paths])
        
        # Print training summary
        print_training_summary(combined_metrics)
        
        # Create plots directory if it doesn't exist
        plots_dir = script_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Create and save the plot
        save_path = plots_dir / "training_validation_loss.png"
        save_path_pdf = plots_dir / "training_validation_loss.pdf"
        
        fig, ax = create_loss_plot(combined_metrics, save_path=str(save_path), show_plot=False)
        
        # Also save as PDF for publications
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"Plot also saved as PDF: {save_path_pdf}")
        
        # Save the combined metrics as CSV for future reference
        csv_path = plots_dir / "combined_training_metrics.csv"
        combined_metrics.to_csv(csv_path, index=False)
        logger.info(f"Combined metrics saved to: {csv_path}")
        
        logger.info("Training loss plotting completed successfully!")
        
    except Exception as e:
        logger.error(f"Error creating training loss plot: {e}")
        raise


if __name__ == "__main__":
    main() 