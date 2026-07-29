"""Entry point: run the pilot grid end-to-end and write results + plots.

    python -m qcore_expr_train.run_pilot

Writes raw per-seed measurements to results/pilot_raw.csv, the seed-aggregated
summary to results/pilot_summary.csv, and four plots to results/plots/.
"""

from __future__ import annotations

from pathlib import Path

from qcore_expr_train.circuits import PILOT_GRID
from qcore_expr_train.results import run_grid, save_results, summarize
from qcore_expr_train.visualize import plot_all

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEEDS = [0, 1, 2]


def main() -> None:
    print(f"Running pilot grid: {len(PILOT_GRID)} cells x {len(SEEDS)} seeds")
    raw = run_grid(PILOT_GRID, seeds=SEEDS)
    save_results(raw, RESULTS_DIR / "pilot_raw.csv")

    summary = summarize(raw)
    save_results(summary, RESULTS_DIR / "pilot_summary.csv")

    plot_all(summary, RESULTS_DIR / "plots")
    print(f"Done. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
