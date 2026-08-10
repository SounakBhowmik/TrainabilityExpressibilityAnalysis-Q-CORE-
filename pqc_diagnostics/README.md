# pqc_diagnostics

Expressibility, entangling capability, trainability, and kernel-usefulness
diagnostics for parameterized quantum circuits (PQCs) — usable on any feature
map, ansatz, or noise backend you bring, not just a specific study's fixed
set.

See `../qcore_expr_train/` (this repo's sibling package) for a worked example
that builds a specific, reproducible research grid on top of this library.

## Install

```bash
pip install .          # from this repo's root
```

## Quickstart: bring your own feature map

```python
from qiskit.circuit import QuantumCircuit, ParameterVector

from pqc_diagnostics.registry import register_feature_map
from pqc_diagnostics.circuits import CircuitSpec
from pqc_diagnostics.metrics import expressibility_kl, gradient_variance, meyer_wallach


def my_feature_map(feature_dimension: int, reps: int, entanglement: str) -> QuantumCircuit:
    x = ParameterVector("x", feature_dimension)
    qc = QuantumCircuit(feature_dimension)
    for _ in range(reps):
        for i in range(feature_dimension):
            qc.rz(x[i], i)
    return qc


register_feature_map("my_fm", my_feature_map)

spec = CircuitSpec(feature_map_family="my_fm", ansatz_family="real_amplitudes",
                    num_qubits=4, depth=2, topology="linear")

print(expressibility_kl(spec.feature_map()))
print(meyer_wallach(spec.feature_map()))
print(gradient_variance(spec.composed_circuit()))
```

`real_amplitudes` and `rotation_only` (ansätze), and `zz`/`local_z`/`cx_ry`
(feature maps) are pre-registered built-ins — `register_feature_map`/
`register_ansatz` (`pqc_diagnostics.registry`) only need to be called for
your *own* additions.

## What's in here

| Module | What it's for |
|---|---|
| `metrics.py` | The 4 measurements: `expressibility_kl`, `meyer_wallach`, `gradient_variance`, `kernel_usefulness`. Each takes a `simulate(circuit, params) -> Statevector \| DensityMatrix` callable, so the same function covers noiseless and noisy simulation. |
| `circuits.py` | `CircuitSpec` (one grid cell: feature map + ansatz + qubits + depth + topology) and `build_grid()` (Cartesian-product sweep builder). |
| `registry.py` | `register_feature_map`/`register_ansatz`/`get_*`/`registered_*` - the extension point for bringing your own PQC. Global, mutable, process-wide singleton (pass explicit family-name lists to `build_grid()` if you need a sweep immune to what else gets registered later in the same process). |
| `evaluation.py` | `evaluate_cell()`/`run_grid()` - generic evaluation pipeline. Supply a `list[MetricSpec]` (what to measure), an optional `data_sample_provider` (real data for trainability's data-binding), and an optional `NoiseProvider` (for the noisy condition). |
| `results.py` | Generic `save_results`/`load_results` (CSV I/O) and `summarize()` (seed-aggregation into mean ± std), parametrized by whatever metric/group columns your `MetricSpec` list produces. |
| `hardware/providers.py` | `NoiseProvider` - transpile onto a fixed qubit patch of any `BackendV2`, then simulate under its calibrated noise. Avoids the "circuit padded out to the backend's full width" blowup that makes naive noisy simulation infeasible. |
| `hardware/registry.py` | `register_provider`/`get_provider` - name a specific `NoiseProvider` configuration for reuse. |
| `visualize/styles.py` | `StyleRegistry` - color/marker/linestyle keyed by `(dimension, name)` (e.g. `("feature_map_family", "my_fm")`), with matplotlib's own cycling as the fallback for anything unregistered. |
| `visualize/plots.py` | `plot_tradeoff`/`plot_metric_vs_depth` - generic plots over a `summarize()`-shaped DataFrame, reading style from a `StyleRegistry` rather than a hardcoded dict. |

## Tests

```bash
python -m pytest pqc_diagnostics/tests
```
