"""Global, mutable registry of feature-map and ansatz circuit factories.

Lets a user plug in their own PQC/feature map without editing this library's
source: `register_feature_map("my_fm", my_factory)` makes "my_fm" usable
anywhere a `feature_map_family` string is accepted (CircuitSpec, build_grid).

This is a process-global singleton (matching the registry convention this
project's qml_pipeline sibling already uses) - simplest ergonomics for a
standalone research/diagnostic tool, at the cost of being shared across
anything importing this module in the same interpreter. If that matters for
your use case (e.g. embedding this in a long-running multi-tenant service),
prefer passing explicit literal family-name lists to build_grid() rather than
relying on "whatever's currently registered" - that's what qcore_expr_train's
own PILOT_GRID/FULL_GRID do.
"""

from __future__ import annotations

from typing import Callable

from qiskit.circuit import QuantumCircuit

FeatureMapFactory = Callable[..., QuantumCircuit]  # (feature_dimension, reps, entanglement) -> QuantumCircuit
AnsatzFactory = Callable[..., QuantumCircuit]  # (num_qubits, reps, entanglement) -> QuantumCircuit

_FEATURE_MAPS: dict[str, FeatureMapFactory] = {}
_ANSATZE: dict[str, AnsatzFactory] = {}


def register_feature_map(name: str, factory: FeatureMapFactory, *, overwrite: bool = False) -> None:
    if not overwrite and name in _FEATURE_MAPS:
        raise ValueError(f"feature map '{name}' is already registered (pass overwrite=True to replace it)")
    _FEATURE_MAPS[name] = factory


def register_ansatz(name: str, factory: AnsatzFactory, *, overwrite: bool = False) -> None:
    if not overwrite and name in _ANSATZE:
        raise ValueError(f"ansatz '{name}' is already registered (pass overwrite=True to replace it)")
    _ANSATZE[name] = factory


def get_feature_map(name: str) -> FeatureMapFactory:
    try:
        return _FEATURE_MAPS[name]
    except KeyError:
        raise KeyError(f"no feature map registered under '{name}' - registered: {registered_feature_maps()}") from None


def get_ansatz(name: str) -> AnsatzFactory:
    try:
        return _ANSATZE[name]
    except KeyError:
        raise KeyError(f"no ansatz registered under '{name}' - registered: {registered_ansatze()}") from None


def registered_feature_maps() -> tuple[str, ...]:
    return tuple(_FEATURE_MAPS)


def registered_ansatze() -> tuple[str, ...]:
    return tuple(_ANSATZE)
