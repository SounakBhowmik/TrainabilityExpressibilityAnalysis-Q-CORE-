"""Entry point for one SLURM array task: evaluate a single grid cell.

    python -m qcore_expr_train.run_full --task-id 0
    python -m qcore_expr_train.run_full   # reads $SLURM_ARRAY_TASK_ID instead

One task = one CircuitSpec from circuits.FULL_GRID, evaluated across all
seeds and both noise conditions, written to its own shard CSV. Run
aggregate.py once every task has finished to combine shards into the full
result set and generate the poster plots. See scripts/submit_full_grid.slurm
for the SLURM array job that drives this at full scale, and the README for
resource/timing guidance.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from qcore_expr_train.circuits import FULL_GRID
from qcore_expr_train.results import evaluate_cell, save_results

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEEDS = list(range(10))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Index into circuits.FULL_GRID. Defaults to $SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "shards")
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    spec = FULL_GRID[task_id]
    print(f"[task {task_id}] evaluating {spec}")

    t0 = time.time()
    df = evaluate_cell(spec, seeds=SEEDS, noise_conditions=("noiseless", "noisy"))
    elapsed = time.time() - t0

    out_path = args.output_dir / f"cell_{task_id:03d}.csv"
    save_results(df, out_path)
    print(f"[task {task_id}] wrote {out_path} ({len(df)} rows) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
