"""This paper's specific noise backend: FakeSherbrooke restricted to a fixed
10-qubit patch, built on pqc_diagnostics's generic NoiseProvider.

See pqc_diagnostics/hardware/providers.py for why the fixed-patch approach is
needed at all (padding to a backend's full width makes density-matrix
simulation infeasible) and why a 4/6/8-qubit circuit safely reuses a prefix
of the same 10-qubit patch a 10-qubit circuit uses.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from pqc_diagnostics.hardware.providers import NoiseProvider

MAX_PATCH_QUBITS = 10  # matches circuits.FULL_GRID's largest qubit count

SHERBROOKE_PATCH = NoiseProvider(
    backend=FakeSherbrooke(),
    max_patch_qubits=MAX_PATCH_QUBITS,
    basis_gates=["sx", "rz", "x", "ecr"],
)


def transpile_for_noise(circuit: QuantumCircuit, seed: int = 0) -> QuantumCircuit:
    return SHERBROOKE_PATCH.transpile_for_noise(circuit, seed=seed)


def noisy_simulate(circuit: QuantumCircuit, params: np.ndarray) -> DensityMatrix:
    return SHERBROOKE_PATCH.noisy_simulate(circuit, params)
