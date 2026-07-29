#!/bin/bash
# Submits circuits.FULL_GRID as a SLURM array job - one task per grid cell.
#
#   sbatch scripts/submit_full_grid.slurm
#
# Adjust --array's upper bound if circuits.FULL_GRID's size changes
# (len(circuits.FULL_GRID) - 1: currently 192 cells, 0-191 - down from the
# earlier 384-cell grid after dropping the pauli feature map, capping qubits
# at 10, and fixing the local_z + rotation_only pair to a single topology
# since it has no entangling gate to route regardless of topology).
#
# --time below is a conservative placeholder, NOT a measured number - dry
# run before trusting it, on this cluster's own hardware (local timing on a
# laptop isn't representative of a compute node):
#   python -m qcore_expr_train.run_full --task-id 0    # cheapest cell (4 qubits, depth 1)
#   python -m qcore_expr_train.run_full --task-id 143  # most expensive cell (10 qubits, depth 8, full topology)
# Multiply task 143's printed per-task time by ~1.3-1.5x for a safety margin,
# and set --time to that. Depth now goes up to 8 (was 4), so the old grid's
# ~29 min/task estimate no longer applies - local measurements this session
# (noiseless only, smaller kernel-sample cap than production) showed the
# depth-8/10-qubit corner costing ~7x the depth-1/10-qubit corner just on the
# noiseless side, before the noisy condition's cost is even added.


#!/bin/bash
#SBATCH --job-name=expr-train-grid
#SBATCH --array=0-191%20

#SBATCH --output=exp-tr.out
#SBATCH --error=exp-tr.err

#SBATCH --gres=gpu:1

#SBATCH --time=2-00:00:00

#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH -A hthapliyal_cy-qubit_0001

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/shards

module load conda

conda activate exp_tr_base
python -m qcore_expr_train.run_full --task-id "$SLURM_ARRAY_TASK_ID"
