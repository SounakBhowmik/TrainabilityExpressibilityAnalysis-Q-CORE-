"""This paper's specific grid-cell schema and evaluation, built on top of the
generic pqc_diagnostics.evaluation/results modules.

evaluate_cell() is the unit of parallelism for the full HPC grid: run_full.py
calls it once per SLURM array task (one CircuitSpec per task). run_grid() is
a thin loop over it, used by the pilot.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pqc_diagnostics import metrics as _metrics
from pqc_diagnostics.evaluation import MetricSpec
from pqc_diagnostics.evaluation import evaluate_cell as _core_evaluate_cell
from pqc_diagnostics.results import load_results, save_results  # noqa: F401 - re-export
from pqc_diagnostics.results import summarize as _core_summarize

from qcore_expr_train.circuits import CircuitSpec
from qcore_expr_train.datasets import prepare_breast_cancer

METRIC_COLUMNS = ["expressibility_kl", "meyer_wallach", "gradient_variance"]
GROUP_COLUMNS = ["feature_map_family", "ansatz_family", "num_qubits", "depth", "topology", "noise_condition"]

# Noisy density-matrix simulation costs ~0.5s per Aer call (measured) regardless
# of qubit count in this grid's range, so its sample counts are cut well below
# the noiseless (exact statevector, cheap) defaults to keep runtime tractable.
_NOISELESS_SAMPLE_COUNTS = {"expressibility": 500, "meyer_wallach": 200, "gradient_variance": 200}
_NOISY_SAMPLE_COUNTS = {"expressibility": 60, "meyer_wallach": 30, "gradient_variance": 30}


def _counts(metric_key: str, condition: str) -> int:
    return (_NOISELESS_SAMPLE_COUNTS if condition == "noiseless" else _NOISY_SAMPLE_COUNTS)[metric_key]


METRICS = [
    MetricSpec(
        "expressibility_kl", "feature_map",
        bind=lambda c, seed, condition, simulate: _metrics.expressibility_kl(
            c, num_samples=_counts("expressibility", condition), seed=seed, simulate=simulate
        ),
    ),
    MetricSpec(
        "meyer_wallach", "feature_map",
        bind=lambda c, seed, condition, simulate: _metrics.meyer_wallach(
            c, num_samples=_counts("meyer_wallach", condition), seed=seed, simulate=simulate
        ),
    ),
    MetricSpec(
        "gradient_variance", "composed",
        bind=lambda c, seed, condition, simulate: _metrics.gradient_variance(
            c, num_samples=_counts("gradient_variance", condition), seed=seed, simulate=simulate
        ),
    ),
]


def evaluate_cell(
    spec: CircuitSpec,
    seeds: list[int],
    noise_conditions: tuple[str, ...] = ("noiseless",),
) -> pd.DataFrame:
    """Evaluate one grid cell across all seeds and noise conditions."""
    noise_provider = None
    if "noisy" in noise_conditions:
        # Deferred import: only pulls in qiskit-aer/qiskit-ibm-runtime when a
        # noisy condition is actually requested, so the noiseless-only pilot
        # never needs those dependencies installed.
        from qcore_expr_train.noise import SHERBROOKE_PATCH

        noise_provider = SHERBROOKE_PATCH

    return _core_evaluate_cell(
        spec,
        seeds,
        METRICS,
        noise_conditions=noise_conditions,
        noise_provider=noise_provider,
        data_sample_provider=lambda n: prepare_breast_cancer(n).X_train,
    )


def run_grid(
    grid: list[CircuitSpec],
    seeds: list[int],
    noise_conditions: tuple[str, ...] = ("noiseless",),
) -> pd.DataFrame:
    """Evaluate every grid cell and return a tidy long-format DataFrame - one
    row per (cell, seed, noise condition) measurement. Used by the pilot,
    which runs the whole (small) grid in one process; the full HPC grid uses
    evaluate_cell() directly, one cell per SLURM array task, via run_full.py.
    """
    frames = [evaluate_cell(spec, seeds, noise_conditions=noise_conditions) for spec in grid]
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw per-seed rows into mean +/- std per grid cell, the form
    used for the confidence-interval plots in visualize.py."""
    return _core_summarize(df, METRIC_COLUMNS, GROUP_COLUMNS)
