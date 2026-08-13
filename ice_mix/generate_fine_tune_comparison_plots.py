from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("generate_fine_tune_comparison_plots")


PROJECT_PAIRS = (
    ("IceMix", "IceMix-FineTune-LBFGS"),
    ("IceMix-Augmented-Rotation", "IceMix-Augmented-Rotation-FineTune-LBFGS"),
    ("IceMix-Drop", "IceMix-Drop-FineTune-LBFGS"),
    (
        "IceMix-Drop-Augmented-Rotation",
        "IceMix-Drop-Augmented-Rotation-FineTune-LBFGS",
    ),
)
MODES = ("all", "tracks", "cascades")
ROTATION_COMPARISON_PROJECTS = (
    "IceMix-Augmented-Rotation",
    "IceMix-Augmented-Rotation-FineTune-LBFGS",
    "IceMix-Augmented-Rotation-AdamWEMA-LR2e-6",
    "IceMix-Augmented-Rotation-AdamWEMA-LR6p25e-6",
    "IceMix-Augmented-Rotation-AdamWEMA-LR2e-5",
)


@dataclass(frozen=True)
class RunResult:
    project: str
    job_id: int
    result_csv: str

    @property
    def display_id(self) -> str:
        return f"job {self.job_id}"


def find_runs(base_dir: str) -> Dict[str, List[RunResult]]:
    """Find flat-format prediction results grouped by exact project name."""
    pattern = os.path.join(base_dir, "**", "predictions", "results.csv")
    grouped: Dict[str, List[RunResult]] = {}

    for result_csv in sorted(glob.glob(pattern, recursive=True)):
        run_dir = os.path.dirname(os.path.dirname(result_csv))
        run_name = os.path.basename(run_dir)
        match = re.match(
            r"^(?P<project>.*?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_job-(?P<job_id>\d+)$",
            run_name,
        )
        if match is None:
            logger.debug("Skipping legacy/non-flat run: %s", run_dir)
            continue

        project = match.group("project")
        grouped.setdefault(project, []).append(
            RunResult(
                project=project,
                job_id=int(match.group("job_id")),
                result_csv=result_csv,
            )
        )

    for runs in grouped.values():
        runs.sort(key=lambda run: run.job_id)
    return grouped


def choose_latest(
    grouped: Dict[str, List[RunResult]], project: str
) -> Optional[RunResult]:
    runs = grouped.get(project, [])
    if not runs:
        logger.warning("No prediction results found for %s", project)
        return None
    if len(runs) > 1:
        logger.info(
            "%s has %d prediction results; using the largest job ID (%d)",
            project,
            len(runs),
            runs[-1].job_id,
        )
    return runs[-1]


def select_pairs(
    grouped: Dict[str, List[RunResult]],
) -> List[Tuple[RunResult, RunResult]]:
    selected = []
    for base_project, fine_tuned_project in PROJECT_PAIRS:
        base_run = choose_latest(grouped, base_project)
        fine_tuned_run = choose_latest(grouped, fine_tuned_project)
        if base_run is None or fine_tuned_run is None:
            logger.warning("Skipping incomplete pair for %s", base_project)
            continue
        logger.info(
            "%s: base %s, fine-tuned %s",
            base_project,
            base_run.display_id,
            fine_tuned_run.display_id,
        )
        selected.append((base_run, fine_tuned_run))
    return selected


def required_columns() -> List[str]:
    return [
        "azimuth",
        "zenith",
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
        "pos_x_pred",
        "pos_y_pred",
        "pos_z_pred",
        "position_x",
        "position_y",
        "position_z",
        "energy",
        "pid",
        "interaction_type",
    ]


def load_results(run: RunResult) -> pd.DataFrame:
    logger.info("Loading %s", run.result_csv)
    df = pd.read_csv(run.result_csv)
    missing = [column for column in required_columns() if column not in df.columns]
    if missing:
        raise ValueError(f"{run.result_csv} is missing required columns: {missing}")
    return df


def filter_data_by_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df.copy()
    is_track = (df["pid"].abs() == 14) & (df["interaction_type"] == 1)
    if mode == "tracks":
        return df[is_track].copy()
    if mode == "cascades":
        return df[~is_track].copy()
    raise ValueError(f"Unknown mode: {mode}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def plot_metric(
    base_run: RunResult,
    fine_tuned_run: RunResult,
    base_df: pd.DataFrame,
    fine_tuned_df: pd.DataFrame,
    metric_func: Callable[[pd.DataFrame], np.ndarray],
    ylabel: str,
    title_prefix: str,
    output_dir: str,
    file_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    from plot_utils import compute_statistics

    for mode in MODES:
        logger.info("Plotting %s %s (%s)", base_run.project, title_prefix, mode)
        base_mode = filter_data_by_mode(base_df, mode)
        fine_tuned_mode = filter_data_by_mode(fine_tuned_df, mode)

        if base_mode.empty or fine_tuned_mode.empty:
            logger.warning("Skipping %s: no events after filtering", mode)
            continue

        base_mode["metric"] = metric_func(base_mode)
        fine_tuned_mode["metric"] = metric_func(fine_tuned_mode)
        base_stats = compute_statistics(base_mode, "metric")
        fine_tuned_stats = compute_statistics(fine_tuned_mode, "metric")
        if not base_stats or not fine_tuned_stats:
            logger.warning("Skipping %s: could not compute statistics", mode)
            continue

        centers = np.asarray(base_stats["centers"])
        base_median = np.asarray(base_stats["median"], dtype=float)
        fine_tuned_median = np.asarray(fine_tuned_stats["median"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = fine_tuned_median / base_median

        fig, (ax_main, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(11, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        ax_main.plot(
            centers,
            base_median,
            marker="o",
            label=f"Base ({base_run.display_id})",
        )
        ax_main.fill_between(
            centers, base_stats["lower"], base_stats["upper"], alpha=0.15
        )
        ax_main.plot(
            fine_tuned_stats["centers"],
            fine_tuned_median,
            marker="s",
            label=f"Fine-tuned ({fine_tuned_run.display_id})",
        )
        ax_main.fill_between(
            fine_tuned_stats["centers"],
            fine_tuned_stats["lower"],
            fine_tuned_stats["upper"],
            alpha=0.15,
        )
        ax_main.set_ylabel(ylabel)
        ax_main.set_title(f"{base_run.project}: {title_prefix} - {mode.capitalize()}")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        ax_main.set_yscale("log")

        ax_ratio.plot(centers, ratio, marker="o", color="black")
        ax_ratio.axhline(1.0, color="gray", linestyle="--")
        ax_ratio.set_ylabel("Fine-tuned / Base")
        ax_ratio.set_xlabel("log10(Energy [GeV])")
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.set_ylim(0.5, 1.5)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{file_prefix}_{mode}.png")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        logger.info("Saved %s", output_path)


def plot_pair(base_run: RunResult, fine_tuned_run: RunResult, output_root: str) -> None:
    from plot_utils import calculate_angular_difference, calculate_vertex_distance

    output_dir = os.path.join(output_root, safe_name(base_run.project))
    os.makedirs(output_dir, exist_ok=True)
    base_df = load_results(base_run)
    fine_tuned_df = load_results(fine_tuned_run)

    plot_metric(
        base_run,
        fine_tuned_run,
        base_df,
        fine_tuned_df,
        lambda d: calculate_angular_difference(
            d["azimuth"],
            d["zenith"],
            d["dir_x_pred"],
            d["dir_y_pred"],
            d["dir_z_pred"],
        ),
        "Angular Error [deg]",
        "Angular Resolution",
        output_dir,
        "fine_tune_angular_res",
    )
    plot_metric(
        base_run,
        fine_tuned_run,
        base_df,
        fine_tuned_df,
        lambda d: calculate_vertex_distance(d),
        "Vertex Error [m]",
        "Vertex Resolution",
        output_dir,
        "fine_tune_vertex_res",
    )


def plot_rotation_candidates(
    grouped: Dict[str, List[RunResult]], output_root: str
) -> None:
    """Compare base, LBFGS, and all AdamW+EMA rotation candidates."""
    from plot_utils import (
        calculate_angular_difference,
        calculate_vertex_distance,
        compute_statistics,
    )
    import matplotlib.pyplot as plt

    runs = [choose_latest(grouped, project) for project in ROTATION_COMPARISON_PROJECTS]
    if any(run is None for run in runs):
        logger.warning(
            "Skipping rotation pilot comparison: at least one run is missing"
        )
        return
    selected = [run for run in runs if run is not None]
    frames = {run.project: load_results(run) for run in selected}
    labels = {
        ROTATION_COMPARISON_PROJECTS[0]: "Base",
        ROTATION_COMPARISON_PROJECTS[1]: "LBFGS",
        ROTATION_COMPARISON_PROJECTS[2]: "AdamW+EMA 2e-6",
        ROTATION_COMPARISON_PROJECTS[3]: "AdamW+EMA 6.25e-6",
        ROTATION_COMPARISON_PROJECTS[4]: "AdamW+EMA 2e-5",
    }
    metrics = (
        (
            "angular_res",
            "Angular Resolution",
            "Angular Error [deg]",
            lambda d: calculate_angular_difference(
                d["azimuth"],
                d["zenith"],
                d["dir_x_pred"],
                d["dir_y_pred"],
                d["dir_z_pred"],
            ),
        ),
        (
            "vertex_res",
            "Vertex Resolution",
            "Vertex Error [m]",
            calculate_vertex_distance,
        ),
    )
    output_dir = os.path.join(output_root, "rotation_adamw_ema_pilot")
    os.makedirs(output_dir, exist_ok=True)

    for file_prefix, title, ylabel, metric_func in metrics:
        for mode in MODES:
            stats = {}
            for run in selected:
                frame = filter_data_by_mode(frames[run.project], mode)
                frame["metric"] = metric_func(frame)
                stats[run.project] = compute_statistics(frame, "metric")
            if any(not value for value in stats.values()):
                logger.warning("Skipping %s/%s: statistics unavailable", title, mode)
                continue

            base_stats = stats[ROTATION_COMPARISON_PROJECTS[0]]
            base_median = np.asarray(base_stats["median"], dtype=float)
            fig, (ax_main, ax_ratio) = plt.subplots(
                2,
                1,
                figsize=(11, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            for index, run in enumerate(selected):
                result = stats[run.project]
                centers = np.asarray(result["centers"])
                median = np.asarray(result["median"], dtype=float)
                label = f"{labels[run.project]} ({run.display_id})"
                ax_main.plot(centers, median, marker="o", label=label)
                ax_main.fill_between(
                    centers, result["lower"], result["upper"], alpha=0.10
                )
                if index:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = median / base_median
                    ax_ratio.plot(centers, ratio, marker="o", label=labels[run.project])

            ax_main.set_ylabel(ylabel)
            ax_main.set_title(f"Rotation Fine-Tuning: {title} - {mode.capitalize()}")
            ax_main.set_yscale("log")
            ax_main.grid(True, alpha=0.3)
            ax_main.legend()
            ax_ratio.axhline(1.0, color="gray", linestyle="--")
            ax_ratio.set_ylabel("Candidate / Base")
            ax_ratio.set_xlabel("log10(Energy [GeV])")
            ax_ratio.set_ylim(0.5, 1.5)
            ax_ratio.grid(True, alpha=0.3)
            ax_ratio.legend(fontsize="small")
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{file_prefix}_{mode}.png")
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            logger.info("Saved %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare each IceMix base model with its LBFGS fine-tuned model."
    )
    parser.add_argument(
        "--base-dir",
        default="ice_mix/outputs",
        help="Base output directory containing run folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for plots. Defaults to " "<base-dir>/fine_tune_comparison_plots."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the selected run pairs."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        raise FileNotFoundError(f"Base directory does not exist: {args.base_dir}")

    grouped = find_runs(args.base_dir)
    selected = select_pairs(grouped)
    if args.dry_run:
        return
    if not selected:
        logger.warning("No complete base/fine-tuned pairs found; no plots generated")
        return

    from plot_utils import setup_matplotlib_style

    setup_matplotlib_style()
    output_root = args.output_dir or os.path.join(
        args.base_dir, "fine_tune_comparison_plots"
    )
    os.makedirs(output_root, exist_ok=True)
    for base_run, fine_tuned_run in selected:
        plot_pair(base_run, fine_tuned_run, output_root)
    plot_rotation_candidates(grouped, output_root)

    logger.info("Done. Plots saved under %s", output_root)


if __name__ == "__main__":
    main()
