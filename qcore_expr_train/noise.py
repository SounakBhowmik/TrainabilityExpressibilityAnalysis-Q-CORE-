"""One consistent noisy-simulation backend, reused across every grid cell.

Rejected approach: `transpile(circuit, backend=some_fake_backend)` pads the
circuit out to the *backend's* full qubit count (verified: a 4-qubit circuit
transpiled against FakeSherbrooke's 127 qubits becomes a 127-qubit-wide
circuit with 123 idle qubits) - density-matrix simulation of that is
infeasible (scales as 4^n). Using a different, smaller fake backend per qubit
count avoids that blowup but confounds "qubit count" with "which chip".

What works instead: pick ONE real backend (FakeSherbrooke, a 127-qubit
Eagle r3 device) and restrict every circuit to a fixed, nested set of its
lowest-index qubits (0..9, the max qubit count anywhere in the grid). Those
qubits happen to form a connected chain on Sherbrooke's heavy-hex coupling
map, so a 4/6/8-qubit circuit just uses a prefix of the same 10 qubits and
edges a 10-qubit circuit uses - one backend, one fixed hardware patch, for
every cell in the grid. The noise model is built once from the full backend
and used unmodified: Aer only applies noise entries for qubit indices that
actually appear in the circuit being run, so a circuit confined to qubits
0-9 correctly picks up only the real calibrated noise for that patch.
"""

from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

MAX_PATCH_QUBITS = 10  # matches circuits.FULL_GRID's largest qubit count

_backend = FakeSherbrooke()
_noise_model = NoiseModel.from_backend(_backend)
_basis_gates = ["sx", "rz", "x", "ecr"]
_patch_edges = [
    (u, v)
    for u, v in _backend.coupling_map.get_edges()
    if u < MAX_PATCH_QUBITS and v < MAX_PATCH_QUBITS
]
_simulator = AerSimulator(method="density_matrix", noise_model=_noise_model)


def transpile_for_noise(circuit: QuantumCircuit, seed: int = 0) -> QuantumCircuit:
    """Transpile a (still-parameterized) circuit onto the fixed qubit patch.

    Transpiling before binding parameters means this only needs to happen
    once per (CircuitSpec, role) - not once per random parameter draw or
    per seed - since routing/optimization doesn't depend on the bound values.
    A fixed seed_transpiler keeps the routing decision (and thus gate counts)
    reproducible across runs.
    """
    n = circuit.num_qubits
    if n > MAX_PATCH_QUBITS:
        raise ValueError(f"circuit has {n} qubits, exceeds the fixed {MAX_PATCH_QUBITS}-qubit patch")
    sub_edges = [(u, v) for u, v in _patch_edges if u < n and v < n]
    return transpile(
        circuit,
        coupling_map=CouplingMap(sub_edges),
        basis_gates=_basis_gates,
        optimization_level=3,
        seed_transpiler=seed,
    )


def noisy_simulate(circuit: QuantumCircuit, params: np.ndarray) -> DensityMatrix:
    """Bind `params` onto an already-transpiled circuit and return its
    density matrix under the fixed patch's calibrated noise. Matches the
    `simulate(circuit, params) -> Statevector | DensityMatrix` signature
    metrics.py's measurement functions expect."""
    bound = circuit.assign_parameters(params)
    bound.save_density_matrix()
    result = _simulator.run(bound).result()
    return DensityMatrix(result.data()["density_matrix"])
