"""Generic per-grid-cell evaluation pipeline: build circuits from a spec, run
them through each requested noise condition, call arbitrary metric functions,
and collect one row per (seed, noise condition).

The experiment-specific pieces a study bolts on - which metrics to compute
and at what sample count, where real data samples come from (if the composed
circuit needs its feature-map half bound to real data rather than staying
free), and which noise backend to use - are all supplied by the caller via
MetricSpec / a data-sample-provider callable / a NoiseProvider, rather than
hardcoded here, so this module has no dependency on any particular dataset
or study.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import numpy as np
import pandas as pd

from pqc_diagnostics.circuits import CircuitSpec
from pqc_diagnostics.hardware.providers import NoiseProvider
from pqc_diagnostics.metrics import Simulate, _default_simulate

DataSampleProvider = Callable[[int], np.ndarray]  # (num_qubits) -> array[n_samples, num_qubits]


class MetricSpec:
    """One measurement to compute per (seed, noise condition).

    `bind(circuit, seed, condition, simulate) -> float` is called with
    `circuit` set to the feature-map-only circuit if `on == "feature_map"`,
    or the composed (and, if a data_sample_provider was given, data-bound)
    circuit if `on == "composed"`. `condition` ("noiseless"/"noisy") is
    passed through so a caller's `bind` can vary e.g. sample counts by
    condition; `simulate` is always the correct callable for that condition
    (the noiseless default, or a NoiseProvider's `noisy_simulate`).
    """

    def __init__(self, name: str, on: str, bind: Callable[..., float]):
        if on not in ("feature_map", "composed"):
            raise ValueError(f"on must be 'feature_map' or 'composed', got {on!r}")
        self.name = name
        self.on = on
        self.bind = bind


def evaluate_cell(
    spec: CircuitSpec,
    seeds: list[int],
    metrics: list[MetricSpec],
    *,
    noise_conditions: tuple[str, ...] = ("noiseless",),
    noise_provider: NoiseProvider | None = None,
    data_sample_provider: DataSampleProvider | None = None,
) -> pd.DataFrame:
    """Evaluate one grid cell across all seeds and noise conditions."""
    feature_map = spec.feature_map()
    ansatz = spec.ansatz()
    # Feature map + ansatz composed on the same qubits - what a real VQC
    # trains on. Built from the same feature_map/ansatz instances used below,
    # so their Parameter objects (needed to bind a data sample further down)
    # are the same objects that appear inside `composed` - qiskit's
    # compose() doesn't copy them.
    composed = feature_map.compose(ansatz)

    data_samples = data_sample_provider(spec.num_qubits) if data_sample_provider is not None else None

    # (condition, feature_map_circuit, composed_circuit, simulate) - built
    # once per spec, reused across every seed, since neither transpilation
    # (noisy path) nor the circuits themselves depend on the seed.
    conditions: list[tuple[str, object, object, Simulate]] = []
    if "noiseless" in noise_conditions:
        conditions.append(("noiseless", feature_map, composed, _default_simulate))
    if "noisy" in noise_conditions:
        if noise_provider is None:
            raise ValueError("noise_provider is required when 'noisy' is in noise_conditions")
        noisy_fm = noise_provider.transpile_for_noise(feature_map)
        noisy_composed = noise_provider.transpile_for_noise(composed)
        conditions.append(("noisy", noisy_fm, noisy_composed, noise_provider.noisy_simulate))

    spec_fields = asdict(spec)
    rows: list[dict] = []
    for seed in seeds:
        for condition, fm, composed_circuit, simulate in conditions:
            if data_samples is not None:
                # Bind the feature map's half of the composed circuit to a
                # real data sample (cycling through the sample pool by seed),
                # leaving only the ansatz's parameters free - so trainability
                # reflects training on an actual encoded data point, not an
                # arbitrary state.
                data_sample = data_samples[seed % len(data_samples)]
                trainable_circuit = composed_circuit.assign_parameters(
                    dict(zip(feature_map.parameters, data_sample))
                )
            else:
                trainable_circuit = composed_circuit

            row = dict(spec_fields, seed=seed, noise_condition=condition)
            for metric in metrics:
                circuit = fm if metric.on == "feature_map" else trainable_circuit
                row[metric.name] = metric.bind(circuit, seed, condition, simulate)
            rows.append(row)

    return pd.DataFrame(rows)


def run_grid(
    grid: list[CircuitSpec],
    seeds: list[int],
    metrics: list[MetricSpec],
    *,
    noise_conditions: tuple[str, ...] = ("noiseless",),
    noise_provider: NoiseProvider | None = None,
    data_sample_provider: DataSampleProvider | None = None,
) -> pd.DataFrame:
    """Evaluate every grid cell and return a tidy long-format DataFrame - one
    row per (cell, seed, noise condition) measurement."""
    frames = [
        evaluate_cell(
            spec, seeds, metrics,
            noise_conditions=noise_conditions,
            noise_provider=noise_provider,
            data_sample_provider=data_sample_provider,
        )
        for spec in grid
    ]
    return pd.concat(frames, ignore_index=True)
