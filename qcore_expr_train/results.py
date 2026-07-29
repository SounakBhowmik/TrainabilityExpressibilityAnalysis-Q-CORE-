"""Runs the circuit grid end-to-end and persists raw per-(spec, seed) results.

Each row is one measurement of one grid cell at one seed and noise condition -
aggregation (mean +/- std across seeds, for confidence intervals) happens
later via summarize(), so the raw data stays available for whatever grouping
an analysis needs.

evaluate_cell() is the unit of parallelism for the full HPC grid: run_full.py
calls it once per SLURM array task (one CircuitSpec per task). run_grid() is
a thin loop over it, used by the pilot (which stays noiseless-only, so it
never needs the qiskit-aer/qiskit-ibm-runtime dependencies noise.py pulls in).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from qcore_expr_train.circuits import CircuitSpec
from qcore_expr_train.datasets import prepare_breast_cancer
from qcore_expr_train.metrics import (
    expressibility_kl,
    gradient_variance,
    meyer_wallach,
)

METRIC_COLUMNS = ["expressibility_kl", "meyer_wallach", "gradient_variance"]
GROUP_COLUMNS = ["feature_map_family", "ansatz_family", "num_qubits", "depth", "topology", "noise_condition"]

# Noisy density-matrix simulation costs ~0.5s per Aer call (measured) regardless
# of qubit count in this grid's range, so its sample counts are cut well below
# the noiseless (exact statevector, cheap) defaults to keep runtime tractable.
_NOISELESS_SAMPLE_COUNTS = {"expressibility": 500, "meyer_wallach": 200, "gradient_variance": 200}
_NOISY_SAMPLE_COUNTS = {"expressibility": 60, "meyer_wallach": 30, "gradient_variance": 30}


@dataclass
class GridCellResult:
    feature_map_family: str
    ansatz_family: str
    num_qubits: int
    depth: int
    topology: str
    seed: int
    noise_condition: str
    expressibility_kl: float
    meyer_wallach: float
    gradient_variance: float


def evaluate_cell(
    spec: CircuitSpec,
    seeds: list[int],
    noise_conditions: tuple[str, ...] = ("noiseless",),
) -> pd.DataFrame:
    """Evaluate one grid cell across all seeds and noise conditions."""
    bundle = prepare_breast_cancer(spec.num_qubits)
    X_train = bundle.X_train

    feature_map = spec.feature_map()
    ansatz = spec.ansatz()
    # Feature map + ansatz composed on the same qubits - what a real VQC
    # trains on. Gradient variance is measured on *this*, not the ansatz
    # alone: the ansatz in isolation ignores the encoding's own contribution
    # to the circuit a VQC actually has to train through. Built from the same
    # feature_map/ansatz instances used below, so their Parameter objects
    # (needed to bind the data half further down) are the same objects that
    # appear inside `composed` - qiskit's compose() doesn't copy them.
    composed = feature_map.compose(ansatz)

    # (condition, feature_map, composed_circuit, sample_counts, simulate_kwargs)
    # - built once per spec, reused across every seed, since neither
    # transpilation (noisy path) nor the circuits themselves depend on the seed.
    conditions = []
    if "noiseless" in noise_conditions:
        conditions.append(("noiseless", feature_map, composed, _NOISELESS_SAMPLE_COUNTS, {}))
    if "noisy" in noise_conditions:
        # Deferred import: only pulls in qiskit-aer/qiskit-ibm-runtime when a
        # noisy condition is actually requested, so the noiseless-only pilot
        # never needs those dependencies installed.
        from qcore_expr_train import noise as noise_backend

        noisy_fm = noise_backend.transpile_for_noise(feature_map)
        noisy_composed = noise_backend.transpile_for_noise(composed)
        conditions.append(
            ("noisy", noisy_fm, noisy_composed, _NOISY_SAMPLE_COUNTS, {"simulate": noise_backend.noisy_simulate})
        )

    rows: list[GridCellResult] = []
    for seed in seeds:
        for condition, fm, composed_circuit, counts, sim_kwargs in conditions:
            # Bind the feature map's half of the composed circuit to a real
            # data sample (cycling through the training set by seed), leaving
            # only the ansatz's parameters free - so trainability reflects
            # training on an actual encoded data point, not an arbitrary state.
            data_sample = X_train[seed % len(X_train)]
            trainable_circuit = composed_circuit.assign_parameters(dict(zip(feature_map.parameters, data_sample)))

            rows.append(
                GridCellResult(
                    feature_map_family=spec.feature_map_family,
                    ansatz_family=spec.ansatz_family,
                    num_qubits=spec.num_qubits,
                    depth=spec.depth,
                    topology=spec.topology,
                    seed=seed,
                    noise_condition=condition,
                    expressibility_kl=expressibility_kl(fm, num_samples=counts["expressibility"], seed=seed, **sim_kwargs),
                    meyer_wallach=meyer_wallach(fm, num_samples=counts["meyer_wallach"], seed=seed, **sim_kwargs),
                    gradient_variance=gradient_variance(
                        trainable_circuit, num_samples=counts["gradient_variance"], seed=seed, **sim_kwargs
                    ),
                )
            )

    return pd.DataFrame([asdict(r) for r in rows])


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


def save_results(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_results(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw per-seed rows into mean +/- std per grid cell, the form
    used for the confidence-interval plots in visualize.py."""
    grouped = df.groupby(GROUP_COLUMNS)[METRIC_COLUMNS]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
    return summary.reset_index()
