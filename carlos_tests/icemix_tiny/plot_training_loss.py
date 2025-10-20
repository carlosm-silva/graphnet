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
import os
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
        DataFrame with columns: epoch, train_loss, val_loss, lr
    """
    logger.info(f"Parsing metrics from: {filepath}")
    
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Separate training, validation loss, and learning rate entries
    train_entries = df[df['train_loss'].notna()][['epoch', 'train_loss']].copy()
    val_entries = df[df['val_loss'].notna()][['epoch', 'val_loss']].copy()
    lr_entries = df[df['lr'].notna()][['epoch', 'lr']].copy()
    
    # Group by epoch to get one value per epoch
    train_by_epoch = train_entries.groupby('epoch').agg({
        'train_loss': 'last'
    }).reset_index()
    val_by_epoch = val_entries.groupby('epoch').agg({
        'val_loss': 'last'
    }).reset_index()
    lr_by_epoch = lr_entries.groupby('epoch').agg({
        'lr': 'last'
    }).reset_index()
    
    # Merge training, validation, and learning rate data
    metrics = pd.merge(train_by_epoch, val_by_epoch, on='epoch', how='outer')
    metrics = pd.merge(metrics, lr_by_epoch, on='epoch', how='outer').sort_values('epoch')
    
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


def create_loss_plot(metrics: pd.DataFrame, save_path: Optional[str] = None, show_plot: bool = True, title_suffix: str = "") -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Create a comprehensive loss vs epoch plot with learning rate subplot.
    
    Args:
        metrics: DataFrame with epoch, train_loss, val_loss, lr columns
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        title_suffix: Optional suffix to add to the plot title
        
    Returns:
        Tuple of (figure, axes) objects
    """
    logger.info("Creating loss vs epoch plot with learning rate subplot...")
    
    # Create figure with two subplots: main loss plot and learning rate subplot
    fig: Figure
    ax_loss: Axes
    ax_lr: Axes
    
    # Create subplots with different heights (loss plot larger than lr plot)
    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(14, 10), dpi=300, 
                                         gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot training loss on main subplot
    train_mask = metrics['train_loss'].notna()
    if train_mask.any():
        ax_loss.plot(metrics.loc[train_mask, 'epoch'], 
                    metrics.loc[train_mask, 'train_loss'], 
                    color=TRAIN_COLOR, 
                    linewidth=2.5, 
                    marker='o', 
                    markersize=6,
                    label='Training Loss',
                    alpha=0.9)
    
    # Plot validation loss on main subplot
    val_mask = metrics['val_loss'].notna()
    if val_mask.any():
        val_data = metrics.loc[val_mask]
        ax_loss.plot(val_data['epoch'], 
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
        ax_loss.plot(min_val_epoch, min_val_loss, 
                    color=MIN_VAL_COLOR, 
                    marker='*', 
                    markersize=15,
                    label=f'Min Val Loss (Epoch {min_val_epoch})',
                    markeredgecolor='black',
                    markeredgewidth=1,
                    zorder=10)
    
    # Customize the main loss plot
    ax_loss.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
    ax_loss.set_ylabel('Loss (log scale)', fontsize=LABEL_FONT_SIZE)
    ax_loss.set_title(f'Training and Validation Loss vs Epoch{title_suffix}', fontsize=TITLE_FONT_SIZE, pad=20)
    
    # Set y-axis to log scale
    ax_loss.set_yscale('log')
    
    # Add grid for better readability
    ax_loss.grid(True, alpha=0.3, linestyle='--', which='major')
    ax_loss.grid(True, alpha=0.15, linestyle=':', which='minor')
    ax_loss.minorticks_on()
    
    # Add legend
    ax_loss.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Set axis limits with some padding
    if len(metrics) > 0:
        ax_loss.set_xlim(metrics['epoch'].min() - 0.5, metrics['epoch'].max() + 0.5)
        
        # Calculate y-limits considering both losses
        all_losses = pd.concat([metrics['train_loss'].dropna(), metrics['val_loss'].dropna()])
        if len(all_losses) > 0:
            y_min, y_max = all_losses.min(), all_losses.max()
            y_range = y_max - y_min
            ax_loss.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    # Plot learning rate on subplot
    lr_mask = metrics['lr'].notna()
    if lr_mask.any():
        lr_data = metrics.loc[lr_mask]
        ax_lr.plot(lr_data['epoch'], 
                   lr_data['lr'], 
                   color='green', 
                   linewidth=2, 
                   marker='o', 
                   markersize=4,
                   alpha=0.8)
        
        # Highlight learning rate changes
        lr_changes = lr_data['lr'].diff().abs() > 1e-10  # Detect significant changes
        if lr_changes.any():
            change_epochs = lr_data.loc[lr_changes, 'epoch']
            change_lrs = lr_data.loc[lr_changes, 'lr']
            ax_lr.plot(change_epochs, change_lrs, 
                      color='red', 
                      marker='^', 
                      markersize=8,
                      linestyle='none',
                      label='LR Changes',
                      zorder=10)
    
    # Customize the learning rate subplot
    ax_lr.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
    ax_lr.set_ylabel('Learning Rate', fontsize=LABEL_FONT_SIZE)
    ax_lr.set_title('Learning Rate vs Epoch', fontsize=LABEL_FONT_SIZE, pad=10)
    
    # Set y-axis to log scale for learning rate
    ax_lr.set_yscale('log')
    
    # Add grid for better readability
    ax_lr.grid(True, alpha=0.3, linestyle='--', which='major')
    ax_lr.grid(True, alpha=0.15, linestyle=':', which='minor')
    ax_lr.minorticks_on()
    
    # Set axis limits for learning rate subplot
    if len(metrics) > 0:
        ax_lr.set_xlim(metrics['epoch'].min() - 0.5, metrics['epoch'].max() + 0.5)
        
        # Calculate y-limits for learning rate
        lr_values = metrics['lr'].dropna()
        if len(lr_values) > 0:
            lr_min, lr_max = lr_values.min(), lr_values.max()
            
            if np.isclose(lr_min, lr_max): # Handle constant learning rate
                # For constant learning rate, set a small range around it for log scale visibility
                if lr_min > 0:
                    ax_lr.set_ylim(lr_min * 0.9, lr_max * 1.1)
                else: 
                    # Fallback for non-positive lr_min, ensure a positive range for log scale
                    ax_lr.set_ylim(1e-6, 1e-1)  # A reasonable default small positive range
            else:
                # For varying learning rate, apply a small multiplicative padding
                # Ensure lower bound is positive for log scale
                lower_bound = lr_min * 0.8
                if lower_bound <= 0: # If calculated lower bound is non-positive, use a very small positive number
                    lower_bound = 1e-7  # Smallest positive number for log scale
                ax_lr.set_ylim(lower_bound, lr_max * 1.2)
    
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
    
    return fig, (ax_loss, ax_lr)


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
    """Main function to create training loss plots for all alpha values."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Define alpha values and their corresponding paths
    alpha_configs = {
        # "alpha_0_026": {
        #     "paths": [
        #         script_dir / "logs" / "alpha_0_026" / "training_logs" / f"version_{i}" / "metrics.csv"
        #         for i in range(len(os.listdir(script_dir / "logs" / "alpha_0_026" / "training_logs")))  
        #     ],
        #     "title_suffix": " (α = 0.026)"
        # },
        "alpha_0_040": {
            "paths": [
                script_dir / "logs" / "alpha_0_040" / "training_logs" / f"version_{i}" / "metrics.csv"
            for i in [3]     
            ],
            "title_suffix": " (α = 0.040)"
        },  
        # "alpha_0_060": {
        #     "paths": [
        #         script_dir / "logs" / "alpha_0_060" / "training_logs" / f"version_{i}" / "metrics.csv"
        #         for i in range(len(os.listdir(script_dir / "logs" / "alpha_0_060" / "training_logs")))
        #     ],
        #     "title_suffix": " (α = 0.060)"
        # },
        #     "full": {
        #     "paths": [
        #         script_dir / "logs" / "training_logs" / f"version_{i}" / "metrics.csv"
        #         for i in [7,8]
        #     ],
        #     "title_suffix": " (Full)"
        # }
    }
    
    # Create plots directory if it doesn't exist
    plots_dir = script_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Process each alpha configuration
    for alpha_name, config in alpha_configs.items():
        logger.info(f"Processing {alpha_name}...")
        
        try:
            # Combine metrics from all versions for this alpha
            combined_metrics = combine_metrics_from_versions([str(p) for p in config["paths"]])
            
            if combined_metrics.empty:
                logger.warning(f"No combined metrics for {alpha_name}. Skipping plot.")
                continue
            
            # Print training summary for this alpha
            logger.info(f"\nTraining Summary for {alpha_name}:")
            print_training_summary(combined_metrics)
            
            # Create and save the plot for this alpha
            save_path = plots_dir / f"{alpha_name}_training_validation_loss.png"
            save_path_pdf = plots_dir / f"{alpha_name}_training_validation_loss.pdf"
            
            # Create plot with custom title
            fig, (ax_loss, ax_lr) = create_loss_plot(
                combined_metrics, 
                save_path=str(save_path), 
                show_plot=False,
                title_suffix=config["title_suffix"]
            )
            
            # Also save as PDF for publications
            plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Plot also saved as PDF: {save_path_pdf}")
            
            # Save the combined metrics as CSV for future reference
            csv_path = plots_dir / f"{alpha_name}_combined_training_metrics.csv"
            combined_metrics.to_csv(csv_path, index=False)
            logger.info(f"Combined metrics saved to: {csv_path}")
            
            plt.close(fig)  # Close the figure to free memory
            
        except Exception as e:
            logger.error(f"Error creating training loss plot for {alpha_name}: {e}")
            continue
    
    logger.info("Training loss plotting completed successfully for all alpha values!")


if __name__ == "__main__":
    main() 