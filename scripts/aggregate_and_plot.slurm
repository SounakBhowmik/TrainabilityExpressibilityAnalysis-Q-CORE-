#!/bin/bash
# Combines submit_full_grid.slurm's shards into results/full_raw.csv,
# results/full_summary.csv, and the 3 poster plots.
#
# Submit after the array job finishes, or chain it directly:
#   ARRAY_ID=$(sbatch --parsable scripts/submit_full_grid.slurm)
#   sbatch --dependency=afterok:$ARRAY_ID scripts/aggregate_and_plot.slurm
#SBATCH --job-name=expr-train-aggregate
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Adjust to your cluster's environment, e.g.:
#   module load python/3.11
#   source .venv/bin/activate

python -m qcore_expr_train.aggregate
