"""Combines run_full.py's per-cell shards into the full result set + poster plots.

    python -m qcore_expr_train.aggregate

Run once every SLURM array task from submit_full_grid.slurm has finished
(scripts/aggregate_and_plot.slurm does this as a dependent follow-up job).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qcore_expr_train.results import save_results, summarize
from qcore_expr_train.visualize import plot_poster_set

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    shard_dir = RESULTS_DIR / "shards"
    shard_paths = sorted(shard_dir.glob("cell_*.csv"))
    if not shard_paths:
        raise SystemExit(f"No shards found in {shard_dir} - run run_full.py (or the SLURM array job) first.")

    print(f"Aggregating {len(shard_paths)} shards from {shard_dir}")
    raw = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
    save_results(raw, RESULTS_DIR / "full_raw.csv")

    summary = summarize(raw)
    save_results(summary, RESULTS_DIR / "full_summary.csv")

    plot_poster_set(summary, RESULTS_DIR / "plots")
    print(f"Done. {len(raw)} rows aggregated, poster plots in {RESULTS_DIR / 'plots'}")


if __name__ == "__main__":
    main()
