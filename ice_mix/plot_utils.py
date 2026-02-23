import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import os

# Constants
ELECTRON_NEUTRINO_PID = 12
MUON_NEUTRINO_PID = 14
TAU_NEUTRINO_PID = 16
CHARGED_CURRENT_INTERACTION = 1
DEFAULT_ENERGY_BINS = 20
LOG_ENERGY_MIN = 1.0
LOG_ENERGY_MAX = 4.0

# Configure logging
logger = logging.getLogger(__name__)


# Plot styling
def setup_matplotlib_style():
    """Configure matplotlib with consistent styling."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except:
        pass

    mpl.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )


def calculate_angular_difference(true_azimuth, true_zenith, pred_x, pred_y, pred_z):
    """Calculate angular difference between true and predicted directions using Haversine formula."""
    true_azimuth = np.asarray(true_azimuth)
    true_zenith = np.asarray(true_zenith)
    pred_x = np.asarray(pred_x)
    pred_y = np.asarray(pred_y)
    pred_z = np.asarray(pred_z)

    pred_azimuth = np.arctan2(pred_y, pred_x)
    pred_zenith = np.arctan2(np.sqrt(pred_x**2 + pred_y**2), pred_z)

    delta_azimuth = pred_azimuth - true_azimuth
    delta_zenith = pred_zenith - true_zenith

    haversine = (
        np.sin(delta_zenith / 2.0) ** 2
        + np.sin(true_zenith) * np.sin(pred_zenith) * np.sin(delta_azimuth / 2.0) ** 2
    )

    haversine = np.clip(haversine, 0.0, 1.0)
    angular_separation = 2.0 * np.arcsin(np.sqrt(haversine))

    return np.degrees(angular_separation)


def calculate_vertex_distance(df, pred_cols=("pos_x_pred", "pos_y_pred", "pos_z_pred")):
    """Calculate Euclidean distance between true and predicted vertex."""
    dist = np.sqrt(
        (df[pred_cols[0]] - df["position_x"]) ** 2
        + (df[pred_cols[1]] - df["position_y"]) ** 2
        + (df[pred_cols[2]] - df["position_z"]) ** 2
    )
    return dist


def load_and_filter_data(csv_path, mode="all"):
    """
    Load data and filter based on mode.
    Modes:
        - "all": All events
        - "tracks": Muon Neutrino Charged Current (pid=14, interaction_type=1)
        - "cascades": Everything else
    """
    if not os.path.exists(csv_path):
        logger.warning(f"File not found: {csv_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading {csv_path}: {e}")
        return pd.DataFrame()

    required_cols = ["event_no", "energy", "pid", "interaction_type"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing column {col} in {csv_path}")
            return pd.DataFrame()

    if mode == "tracks":
        # Numu CC
        df = df[
            (np.abs(df["pid"]) == MUON_NEUTRINO_PID)
            & (df["interaction_type"] == CHARGED_CURRENT_INTERACTION)
        ]
    elif mode == "cascades":
        # Not (Numu CC)
        is_track = (np.abs(df["pid"]) == MUON_NEUTRINO_PID) & (
            df["interaction_type"] == CHARGED_CURRENT_INTERACTION
        )
        df = df[~is_track]

    return df


def compute_statistics(df, value_col, energy_col="energy", n_bins=DEFAULT_ENERGY_BINS):
    """Compute median, 16th, and 84th percentiles in energy bins."""
    if df.empty:
        return None

    df = df.copy()
    df["log_energy"] = np.log10(df[energy_col])

    # Define energy bins
    # Ensure we cover the data range or use defaults
    # min_e = df["log_energy"].min()
    # max_e = df["log_energy"].max()
    # bins = np.linspace(min_e, max_e, n_bins + 1)

    # Fixed bins for consistency across plots
    bins = np.linspace(LOG_ENERGY_MIN, LOG_ENERGY_MAX, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    stats = {"centers": centers, "median": [], "lower": [], "upper": [], "count": []}

    for i in range(len(bins) - 1):
        mask = (df["log_energy"] >= bins[i]) & (df["log_energy"] < bins[i + 1])
        vals = df.loc[mask, value_col]

        if len(vals) > 5:  # Minimum events
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
