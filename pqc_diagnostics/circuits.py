"""Generic PQC building blocks: feature-map/ansatz factories, a grid-cell
spec, and a Cartesian-product grid builder - usable on any registered
feature map/ansatz, not just the 3+2 this library ships with as built-ins.

Topology ("linear"/"circular"/"full") and depth (reps) are handled entirely
by qiskit's functional circuit-library API - CircuitSpec/build_grid just
enumerate combinations and look up factories via the registry, no
per-family circuit-building logic needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import n_local, real_amplitudes, z_feature_map, zz_feature_map

from pqc_diagnostics.registry import (
    get_ansatz,
    get_feature_map,
    register_ansatz,
    register_feature_map,
    registered_ansatze,
    registered_feature_maps,
)

Topology = str  # "linear" | "circular" | "full" - passed straight through to qiskit


def _entangling_pairs(num_qubits: int, entanglement: Topology) -> list[tuple[int, int]]:
    """Qubit pairs for a topology string, matching qiskit's own linear/circular/full
    convention (n_local doesn't expose this pair list as a standalone function)."""
    pairs = [(i, i + 1) for i in range(num_qubits - 1)]
    if entanglement == "linear":
        return pairs
    if entanglement == "circular":
        return pairs + ([(num_qubits - 1, 0)] if num_qubits > 2 else [])
    if entanglement == "full":
        return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
    raise ValueError(f"Unknown entanglement topology '{entanglement}'")


def _cx_ry_factory(feature_dimension: int, reps: int, entanglement: Topology) -> QuantumCircuit:
    # Ry rotations + CX entanglers - a non-diagonal, maximally-entangling
    # feature map, distinct from zz's diagonal RZZ-type entangling gates.
    # Built by hand (rather than n_local) because n_local creates fresh
    # parameters every repetition, whereas a feature map needs the same data
    # parameters reused every repetition - `.repeat()` on a single block does
    # that (same Parameter objects each time), matching how zz_feature_map
    # behaves internally.
    x = ParameterVector("x", feature_dimension)
    block = QuantumCircuit(feature_dimension)
    for i in range(feature_dimension):
        block.ry(x[i], i)
    for i, j in _entangling_pairs(feature_dimension, entanglement):
        block.cx(i, j)
    return block.repeat(reps).decompose()


def _entangling_ansatz(num_qubits: int, reps: int, entanglement: Topology) -> QuantumCircuit:
    return real_amplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)


def _rotation_only_ansatz(num_qubits: int, reps: int, entanglement: Topology) -> QuantumCircuit:
    # No entanglement - `entanglement` is accepted but unused, kept only so
    # every ansatz factory has the same call signature.
    del entanglement
    return n_local(num_qubits=num_qubits, rotation_blocks=["ry"], entanglement_blocks=[], reps=reps)


# Built-in registrations - generic qiskit-circuit-library wrappers (plus the
# hand-rolled cx_ry factory), available under these names as soon as
# pqc_diagnostics is imported. A user can register additional feature
# maps/ansätze under new names via register_feature_map()/register_ansatz()
# without needing to edit this module.
register_feature_map("zz", zz_feature_map)
register_feature_map("local_z", z_feature_map)
register_feature_map("cx_ry", _cx_ry_factory)
register_ansatz("rotation_only", _rotation_only_ansatz)
register_ansatz("real_amplitudes", _entangling_ansatz)


@dataclass(frozen=True)
class CircuitSpec:
    """One grid cell: a feature map and an ansatz, chosen independently, at a
    given width, depth, and topology (shared by both)."""

    feature_map_family: str
    ansatz_family: str
    num_qubits: int
    depth: int
    topology: Topology

    def feature_map(self) -> QuantumCircuit:
        factory = get_feature_map(self.feature_map_family)
        return factory(
            feature_dimension=self.num_qubits, reps=self.depth, entanglement=self.topology
        )

    def ansatz(self) -> QuantumCircuit:
        factory = get_ansatz(self.ansatz_family)
        return factory(self.num_qubits, self.depth, self.topology)

    def composed_circuit(self) -> QuantumCircuit:
        """Feature map followed by ansatz, on the same qubits - what a real
        VQC actually trains on. Gradient variance is measured on this
        composed circuit (see pqc_diagnostics.evaluation), not the ansatz
        alone: the ansatz in isolation would ignore the encoding's own
        contribution to the circuit a VQC actually has to train through, so
        trainability is measured with the feature map's parameters bound to
        a real data sample and only the ansatz's parameters left free."""
        return self.feature_map().compose(self.ansatz())


def build_grid(
    qubits: list[int],
    depths: list[int],
    topologies: list[Topology],
    feature_map_families: list[str] | None = None,
    ansatz_families: list[str] | None = None,
) -> list[CircuitSpec]:
    """Cartesian product of every sweep axis into a flat list of grid cells,
    crossing feature_map_families with ansatz_families rather than pairing
    them 1:1. Omitting feature_map_families/ansatz_families sweeps every
    currently-registered family - resolved at call time, not import time, so
    it reflects whatever's registered when build_grid() actually runs."""
    if feature_map_families is None:
        feature_map_families = registered_feature_maps()
    if ansatz_families is None:
        ansatz_families = registered_ansatze()
    return [
        CircuitSpec(feature_map_family=fm, ansatz_family=an, num_qubits=q, depth=d, topology=t)
        for fm, an, q, d, t in product(feature_map_families, ansatz_families, qubits, depths, topologies)
    ]
