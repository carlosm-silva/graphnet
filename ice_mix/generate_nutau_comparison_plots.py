from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("generate_nutau_comparison_plots")


EXPECTED_PROJECTS = (
    "IceMix-Drop-Augmented-Rotation",
    "IceMix-Augmented-Rotation",
    "IceMix-Drop",
    "IceMix",
)
MODES = ("all", "tracks", "cascades")
LEGACY_ICEMIX_RUN = ("2026-01-30", "22-41-47")


@dataclass(frozen=True)
class RunResult:
    project: str
    job_id: int
    result_csv: str
    run_dir: str
    display_id: str


def find_result_files(base_dir: str) -> List[str]:
    """Find prediction results saved by predict.py."""
    pattern = os.path.join(base_dir, "**", "predictions", "results.csv")
    return sorted(glob.glob(pattern, recursive=True))


def parse_run_result(result_csv: str) -> Optional[RunResult]:
    """Parse project and job id from a flat run directory name.

    Expected directory:
        {project}_{YYYY-MM-DD}_{HH-MM-SS}_job-{job_id}/predictions/results.csv
    """
    run_dir = os.path.dirname(os.path.dirname(result_csv))
    run_name = os.path.basename(run_dir)

    # One-off exception: the original no-nutau IceMix run predates the flat
    # {project}_..._job-{id} naming convention.
    parent_name = os.path.basename(os.path.dirname(run_dir))
    if (parent_name, run_name) == LEGACY_ICEMIX_RUN:
        return RunResult(
            project="IceMix",
            job_id=0,
            result_csv=result_csv,
            run_dir=run_dir,
            display_id=f"{parent_name}_{run_name}",
        )

    match = re.match(
        r"^(?P<project>.*?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_job-(?P<job_id>\d+)$",
        run_name,
    )
    if not match:
        logger.debug("Skipping non-flat run directory: %s", run_dir)
        return None

    return RunResult(
        project=match.group("project"),
        job_id=int(match.group("job_id")),
        result_csv=result_csv,
        run_dir=run_dir,
        display_id=f"job {match.group('job_id')}",
    )


def group_runs(result_files: List[str]) -> Dict[str, List[RunResult]]:
    grouped: Dict[str, List[RunResult]] = {project: [] for project in EXPECTED_PROJECTS}

    for result_csv in result_files:
        parsed = parse_run_result(result_csv)
        if parsed is None:
            continue
        if parsed.project not in grouped:
            logger.debug("Ignoring result from non-target project %s: %s", parsed.project, result_csv)
            continue
        grouped[parsed.project].append(parsed)

    for project in grouped:
        grouped[project].sort(key=lambda r: r.job_id)

    return grouped


def choose_pair(project: str, runs: List[RunResult]) -> Optional[Tuple[RunResult, RunResult]]:
    """Return (without_nutau, with_nutau) based on job id ordering."""
    if len(runs) < 2:
        logger.warning(
            "Skipping %s: expected two runs, found %d result file(s).", project, len(runs)
        )
        return None

    if len(runs) > 2:
        logger.warning(
            "%s has %d result files. Using the two largest job ids: %s",
            project,
            len(runs),
            ", ".join(str(r.job_id) for r in runs[-2:]),
        )

    without_nutau, with_nutau = runs[-2], runs[-1]
    logger.info(
        "%s: No nu_tau %s, With nu_tau %s",
        project,
        without_nutau.display_id,
        with_nutau.display_id,
    )
    return without_nutau, with_nutau


def filter_data_by_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df.copy()
    if mode == "tracks":
        return df[(df["pid"].abs() == 14) & (df["interaction_type"] == 1)].copy()
    if mode == "cascades":
        is_track = (df["pid"].abs() == 14) & (df["interaction_type"] == 1)
        return df[~is_track].copy()
    raise ValueError(f"Unknown mode: {mode}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def required_columns_for_plot() -> List[str]:
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
    import pandas as pd

    logger.info("Loading %s", run.result_csv)
    df = pd.read_csv(run.result_csv)

    missing = [col for col in required_columns_for_plot() if col not in df.columns]
    if missing:
        raise ValueError(f"{run.result_csv} is missing required columns: {missing}")

    return df


def plot_project_metric(
    project: str,
    without_run: RunResult,
    with_run: RunResult,
    without_df: pd.DataFrame,
    with_df: pd.DataFrame,
    metric_func: Callable[[pd.DataFrame], np.ndarray],
    ylabel: str,
    title_prefix: str,
    output_dir: str,
    file_prefix: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from plot_utils import compute_statistics

    for mode in MODES:
        logger.info("Plotting %s %s (%s)", project, title_prefix, mode)

        without_mode = filter_data_by_mode(without_df, mode)
        with_mode = filter_data_by_mode(with_df, mode)

        if without_mode.empty or with_mode.empty:
            logger.warning(
                "Skipping %s %s %s: empty data after filtering. no_nutau=%d with_nutau=%d",
                project,
                title_prefix,
                mode,
                len(without_mode),
                len(with_mode),
            )
            continue

        without_mode["metric"] = metric_func(without_mode)
        with_mode["metric"] = metric_func(with_mode)

        without_stats = compute_statistics(without_mode, "metric")
        with_stats = compute_statistics(with_mode, "metric")

        if not without_stats or not with_stats:
            logger.warning("Skipping %s %s %s: could not compute statistics.", project, title_prefix, mode)
            continue

        centers = np.asarray(without_stats["centers"])
        without_median = np.asarray(without_stats["median"], dtype=float)
        with_median = np.asarray(with_stats["median"], dtype=float)
        ratio = with_median / without_median

        fig, (ax_main, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(11, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        without_label = f"No nu_tau ({without_run.display_id})"
        with_label = f"With nu_tau ({with_run.display_id})"

        ax_main.plot(centers, without_median, marker="o", label=without_label)
        ax_main.fill_between(
            centers,
            without_stats["lower"],
            without_stats["upper"],
            alpha=0.15,
        )

        ax_main.plot(
            with_stats["centers"],
            with_median,
            marker="s",
            label=with_label,
        )
        ax_main.fill_between(
            with_stats["centers"],
            with_stats["lower"],
            with_stats["upper"],
            alpha=0.15,
        )

        ax_main.set_ylabel(ylabel)
        ax_main.set_title(f"{project}: {title_prefix} - {mode.capitalize()}")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        ax_main.set_yscale("log")

        ax_ratio.plot(centers, ratio, marker="o", color="black")
        ax_ratio.axhline(1.0, color="gray", linestyle="--")
        ax_ratio.set_ylabel("With / No")
        ax_ratio.set_xlabel("log10(Energy [GeV])")
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.set_ylim(0.5, 1.5)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{file_prefix}_{mode}.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        logger.info("Saved %s", out_path)


def plot_project(project: str, without_run: RunResult, with_run: RunResult, output_root: str) -> None:
    from plot_utils import calculate_angular_difference, calculate_vertex_distance

    project_output = os.path.join(output_root, safe_name(project))
    os.makedirs(project_output, exist_ok=True)

    without_df = load_results(without_run)
    with_df = load_results(with_run)

    plot_project_metric(
        project=project,
        without_run=without_run,
        with_run=with_run,
        without_df=without_df,
        with_df=with_df,
        metric_func=lambda d: calculate_angular_difference(
            d["azimuth"],
            d["zenith"],
            d["dir_x_pred"],
            d["dir_y_pred"],
            d["dir_z_pred"],
        ),
        ylabel="Angular Error [deg]",
        title_prefix="Angular Resolution",
        output_dir=project_output,
        file_prefix="nutau_angular_res",
    )

    plot_project_metric(
        project=project,
        without_run=without_run,
        with_run=with_run,
        without_df=without_df,
        with_df=with_df,
        metric_func=lambda d: calculate_vertex_distance(d),
        ylabel="Vertex Error [m]",
        title_prefix="Vertex Resolution",
        output_dir=project_output,
        file_prefix="nutau_vertex_res",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate no-nutau vs with-nutau comparison plots for IceMix runs."
    )
    parser.add_argument(
        "--base-dir",
        default="ice_mix/outputs",
        help="Base output directory containing run folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots. Defaults to <base-dir>/nutau_comparison_plots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the selected run pairs.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir or os.path.join(base_dir, "nutau_comparison_plots")

    if not os.path.exists(base_dir):
        logger.error("Base directory does not exist: %s", base_dir)
        return

    result_files = find_result_files(base_dir)
    logger.info("Found %d prediction result file(s).", len(result_files))

    grouped = group_runs(result_files)
    selected: List[Tuple[str, RunResult, RunResult]] = []

    for project in EXPECTED_PROJECTS:
        pair = choose_pair(project, grouped[project])
        if pair is not None:
            selected.append((project, pair[0], pair[1]))

    if args.dry_run:
        return

    if not selected:
        logger.warning("No complete project pairs found. No plots generated.")
        return

    from plot_utils import setup_matplotlib_style

    setup_matplotlib_style()

    os.makedirs(output_dir, exist_ok=True)
    for project, without_run, with_run in selected:
        plot_project(project, without_run, with_run, output_dir)

    logger.info("Done. Plots saved under %s", output_dir)


if __name__ == "__main__":
    main()
