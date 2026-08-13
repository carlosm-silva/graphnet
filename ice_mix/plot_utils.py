"""Share dataframe filtering, reconstruction statistics, and plot styling."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter
import json
import logging
import os
from datetime import datetime, timezone

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


def write_plot_manifest(
    output_dir: str,
    analysis: str,
    input_csvs: List[str],
    reference_csv: Optional[str] = None,
) -> str:
    """Record plot inputs and the current unweighted statistics policy.

    Parameters
    ----------
    output_dir : str
        Existing plot directory receiving ``plot_manifest.json``.
    analysis : str
        Human-readable comparison or plot family name.
    input_csvs : list of str
        Prediction tables used to create the figures.
    reference_csv : str, optional
        Explicit external reference table, when supplied.

    Returns
    -------
    str
        Path to the written JSON manifest.

    Notes
    -----
    Current quantiles give every retained CSV row equal weight. Although
    ``oneweight`` is propagated into prediction tables, its authoritative
    normalization and target population are not established in this package,
    so this function records that it was not used rather than inventing a
    weighted physics convention.
    """
    manifest_path = os.path.join(output_dir, "plot_manifest.json")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": analysis,
        "input_csvs": [os.path.abspath(path) for path in input_csvs],
        "reference_csv": (
            os.path.abspath(reference_csv) if reference_csv is not None else None
        ),
        "weighting": {
            "mode": "unweighted_equal_rows",
            "oneweight_used": False,
            "reason": (
                "IceMix does not define the authoritative oneweight normalization "
                "or target population."
            ),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    return manifest_path


def validate_matching_events(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> None:
    """Require two comparison tables to contain the same event population.

    Parameters
    ----------
    left, right : pandas.DataFrame
        Prediction tables to compare. Identity uses the multiset of
        ``(event_no, pid, interaction_type)`` rows so repeated event numbers
        from different flavor databases remain distinguishable.
    left_name, right_name : str
        Input labels included in validation errors.

    Raises
    ------
    ValueError
        If identity columns are missing or the event multisets differ.
    """
    identity_columns = ["event_no", "pid", "interaction_type"]
    for name, frame in ((left_name, left), (right_name, right)):
        missing = [column for column in identity_columns if column not in frame]
        if missing:
            raise ValueError(f"{name} is missing event identity columns: {missing}")

    def _identity_counts(frame: pd.DataFrame) -> Counter:
        return Counter(
            tuple(values)
            for values in frame[identity_columns].itertuples(index=False, name=None)
        )

    left_counts = _identity_counts(left)
    right_counts = _identity_counts(right)
    if left_counts == right_counts:
        return

    only_left = list((left_counts - right_counts).elements())[:5]
    only_right = list((right_counts - left_counts).elements())[:5]
    raise ValueError(
        "Comparison inputs do not contain identical event populations: "
        f"{left_name} has {len(left)} rows, {right_name} has {len(right)} rows; "
        f"examples only in left={only_left}, only in right={only_right}."
    )


def validate_matching_evaluation_manifests(
    left_results_csv: str,
    right_results_csv: str,
) -> None:
    """Require two prediction tables to record the same evaluation selection.

    Each table must have ``evaluation_manifest.json`` beside it. The comparison
    checks partition name plus the ordered database basenames and exact event-ID
    lists. Checkpoints and study settings may differ because those are normally
    the treatment under comparison.

    Raises
    ------
    FileNotFoundError
        If either manifest is absent.
    ValueError
        If the recorded evaluation populations differ.
    """

    def _load_population(results_csv: str) -> Dict[str, Any]:
        manifest_path = os.path.join(
            os.path.dirname(os.path.abspath(results_csv)),
            "evaluation_manifest.json",
        )
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                "Scientific comparison requires an evaluation manifest beside "
                f"each results.csv; missing {manifest_path}. Regenerate inference "
                "with the current predict.py."
            )
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        return {
            "partition": manifest.get("partition"),
            "datasets": [
                {
                    "basename": os.path.basename(str(dataset.get("path", ""))),
                    "event_no": dataset.get("event_no"),
                }
                for dataset in manifest.get("datasets", [])
            ],
        }

    left_population = _load_population(left_results_csv)
    right_population = _load_population(right_results_csv)
    if left_population != right_population:
        raise ValueError(
            "Comparison manifests describe different partitions or per-database "
            f"event selections: {left_results_csv} versus {right_results_csv}."
        )


# Plot styling
def setup_matplotlib_style():
    """Configure process-global Matplotlib defaults for IceMix figures.

    Notes
    -----
    Mutates ``matplotlib.rcParams``. If the preferred seaborn style is absent,
    the existing style is retained without raising.
    """
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
    """Calculate great-circle separation between truth and prediction.

    Parameters
    ----------
    true_azimuth, true_zenith : array-like
        Truth angles in radians.
    pred_x, pred_y, pred_z : array-like
        Predicted Cartesian direction components.

    Returns
    -------
    numpy.ndarray
        Element-wise angular separation in degrees.
    """
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
    """Calculate Euclidean vertex error in the dataframe coordinate units.

    ``df`` must contain ``position_x/y/z`` and the three columns named by
    ``pred_cols``. Returns one floating-point distance per row. Current plots
    interpret the returned coordinate units as metres.
    """
    dist = np.sqrt(
        (df[pred_cols[0]] - df["position_x"]) ** 2
        + (df[pred_cols[1]] - df["position_y"]) ** 2
        + (df[pred_cols[2]] - df["position_z"]) ** 2
    )
    return dist


def load_and_filter_data(csv_path, mode="all"):
    """Load a prediction CSV and select an event-topology mode.

    Parameters
    ----------
    csv_path : path-like
        Prediction table to read.
    mode : {"all", "tracks", "cascades"}
        Tracks are charged-current muon-neutrino events; cascades are their
        complement.

    Returns
    -------
    pandas.DataFrame
        Selected rows, or an empty frame after missing-file/read/schema errors.
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
    """Compute unweighted binned median and central 68% interval versus log-energy.

    Parameters
    ----------
    df : pandas.DataFrame
        Event table containing the value and positive energy columns.
    value_col : str
        Per-event metric column.
    energy_col : str
        Energy column interpreted as GeV by plot labels.
    n_bins : int
        Number of fixed-width bins between log10 energy 1 and 4.

    Returns
    -------
    dict or None
        Bin centers, median, 16th/84th percentiles, and counts. Bins with five
        or fewer events receive NaN statistics; an empty input returns ``None``.

    Notes
    -----
    Every row contributes equally. Although prediction tables propagate the
    IceCube simulation field ``oneweight``, this function does not consume it;
    all existing plots that call this function are event-count-weighted, not
    population- or flux-weighted. A scientifically defined weighted quantile
    requires an agreed normalization and is intentionally not inferred here.
    """
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
