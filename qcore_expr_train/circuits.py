"""Defines the circuit grid: feature map family x ansatz family x qubit count
x depth x topology.

Each grid cell (a CircuitSpec) crosses one of 3 feature maps with one of 2
ansatze, sharing qubit count/depth/topology between them. The feature map and
ansatz are chosen independently - deliberately, not paired by family - so
that trainability (measured on the two composed together, see results.py) can
be attributed to the feature map's expressibility while the ansatz is held
fixed, rather than confounding "which feature map" with "which ansatz" the
way a fixed 1:1 pairing would.

Topology ("linear"/"circular"/"full") and depth (reps) are handled entirely
by qiskit's functional circuit-library API - the grid just enumerates
combinations and looks up factories, no custom circuit-building logic needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import (
    n_local,
    real_amplitudes,
    z_feature_map,
    zz_feature_map,
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
    # feature map, distinct from zz/pauli's diagonal RZZ-type entangling
    # gates. Built by hand (rather than n_local) because n_local creates
    # fresh parameters every repetition, whereas a feature map needs the same
    # data parameters reused every repetition - `.repeat()` on a single block
    # does that (same Parameter objects each time), matching how
    # zz_feature_map/pauli_feature_map behave internally.
    x = ParameterVector("x", feature_dimension)
    block = QuantumCircuit(feature_dimension)
    for i in range(feature_dimension):
        block.ry(x[i], i)
    for i, j in _entangling_pairs(feature_dimension, entanglement):
        block.cx(i, j)
    return block.repeat(reps).decompose()


_FEATURE_MAPS = {
    "zz": zz_feature_map,
    "local_z": z_feature_map,
    "cx_ry": _cx_ry_factory,
}


def _entangling_ansatz(num_qubits: int, reps: int, entanglement: Topology) -> QuantumCircuit:
    return real_amplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)


def _rotation_only_ansatz(num_qubits: int, reps: int, entanglement: Topology) -> QuantumCircuit:
    # No entanglement - `entanglement` is accepted but unused, kept only so
    # every entry in _ANSATZE has the same call signature.
    del entanglement
    return n_local(num_qubits=num_qubits, rotation_blocks=["ry"], entanglement_blocks=[], reps=reps)


# Exactly 2 ansatze, deliberately the two extremes (no entanglement vs.
# maximal CX entanglement) rather than one bespoke ansatz per feature-map
# family. Each is crossed with every feature map in build_grid() below, so
# the effect of feature-map expressibility on trainability can be isolated
# by holding the ansatz fixed and varying only the feature map - a fixed 1:1
# family-to-ansatz pairing can't support that (any trainability difference
# between families would be confounded with the ansatz also changing).
_ANSATZE = {
    "rotation_only": _rotation_only_ansatz,
    "real_amplitudes": _entangling_ansatz,
}


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
        factory = _FEATURE_MAPS[self.feature_map_family]
        return factory(
            feature_dimension=self.num_qubits, reps=self.depth, entanglement=self.topology
        )

    def ansatz(self) -> QuantumCircuit:
        factory = _ANSATZE[self.ansatz_family]
        return factory(self.num_qubits, self.depth, self.topology)

    def composed_circuit(self) -> QuantumCircuit:
        """Feature map followed by ansatz, on the same qubits - what a real
        VQC actually trains on. Used for trainability (see results.py):
        gradient variance of the ansatz *alone* would ignore the encoding's
        own contribution to the circuit a VQC has to train through, so
        trainability is measured on this composed circuit instead, with the
        feature map's parameters bound to a real data sample and only the
        ansatz's parameters left free."""
        return self.feature_map().compose(self.ansatz())


def build_grid(
    qubits: list[int],
    depths: list[int],
    topologies: list[Topology],
    feature_map_families: list[str] = tuple(_FEATURE_MAPS),
    ansatz_families: list[str] = tuple(_ANSATZE),
) -> list[CircuitSpec]:
    """Cartesian product of every sweep axis into a flat list of grid cells,
    crossing feature_map_families with ansatz_families rather than pairing
    them 1:1."""
    return [
        CircuitSpec(feature_map_family=fm, ansatz_family=an, num_qubits=q, depth=d, topology=t)
        for fm, an, q, d, t in product(feature_map_families, ansatz_families, qubits, depths, topologies)
    ]


# Small pilot sweep to validate the pipeline before scaling to the full grid
# (kernel matrices are O(n^2), and the full grid multiplies qubits x depth x
# topology x feature map x ansatz x noise x seeds on top of that). Qubits and
# depths are deliberately the smallest and largest values in FULL_GRID's own
# sweep (4/10 qubits, depth 1/8) rather than adjacent small values, so the
# pilot also smoke-tests the grid's outer bounds before a full HPC run.
PILOT_GRID = build_grid(
    qubits=[4, 10],
    depths=[1, 8],
    topologies=["linear", "circular"],
)

# Full HPC-scale grid for the poster run - see results.py/run_full.py for how
# this gets sharded one grid cell per SLURM array task. Qubit count is capped
# at 10 to match noise.py's MAX_PATCH_QUBITS (the noisy condition's fixed
# hardware-patch width) - see noise.py for why the noisy condition can't
# currently go wider than that.
_FULL_GRID_QUBITS = [4, 8, 10]
_FULL_GRID_DEPTHS = [1, 2, 4, 8]
_FULL_GRID_TOPOLOGIES = ["linear", "circular", "full"]

# local_z + rotation_only is the one (feature map, ansatz) pair with no
# entangling gate anywhere in the composed circuit (rotation_only discards
# its entanglement argument; z_feature_map has none either) - so topology is
# a no-op for it specifically, and all 3 topologies would evaluate the exact
# same circuit. Every other pair has an entangling gate somewhere (feature
# map, ansatz, or both) and genuinely depends on topology, so only this one
# pair is pulled out and fixed to a single topology instead of crossed with
# all 3 - a free ~11% cut (24 of 216 cells) with no information loss.
FULL_GRID = (
    build_grid(
        qubits=_FULL_GRID_QUBITS,
        depths=_FULL_GRID_DEPTHS,
        topologies=_FULL_GRID_TOPOLOGIES,
        feature_map_families=[fm for fm in _FEATURE_MAPS if fm != "local_z"],
    )
    + build_grid(
        qubits=_FULL_GRID_QUBITS,
        depths=_FULL_GRID_DEPTHS,
        topologies=_FULL_GRID_TOPOLOGIES,
        feature_map_families=["local_z"],
        ansatz_families=[an for an in _ANSATZE if an != "rotation_only"],
    )
    + build_grid(
        qubits=_FULL_GRID_QUBITS,
        depths=_FULL_GRID_DEPTHS,
        topologies=["linear"],
        feature_map_families=["local_z"],
        ansatz_families=["rotation_only"],
    )
)
