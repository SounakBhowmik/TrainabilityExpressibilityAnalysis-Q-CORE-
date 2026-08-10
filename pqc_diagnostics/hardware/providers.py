"""Generic noisy-simulation backend interface.

Rejected approach: `transpile(circuit, backend=some_backend)` pads the
circuit out to the *backend's* full qubit count (verified: a 4-qubit circuit
transpiled against a 127-qubit backend becomes a 127-qubit-wide circuit with
123 idle qubits) - density-matrix simulation of that is infeasible (scales as
4^n). Using a different, smaller backend per qubit count avoids that blowup
but confounds "qubit count" with "which chip".

What works instead: restrict every circuit to a fixed, nested set of a real
backend's lowest-index qubits (0..max_patch_qubits-1). If those qubits form a
connected sub-chain on the backend's coupling map, a narrower circuit just
uses a prefix of the same qubits/edges a wider one uses - one backend, one
fixed hardware patch, for every circuit width up to max_patch_qubits. The
noise model is built once from the full backend and used unmodified: Aer
only applies noise entries for qubit indices that actually appear in the
circuit being run, so a circuit confined to the patch correctly picks up
only the real calibrated noise for that patch.
"""

from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.quantum_info import DensityMatrix
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


class NoiseProvider:
    """One consistent noisy-simulation backend, reused across every circuit
    evaluated against it. Matches the `simulate(circuit, params) ->
    Statevector | DensityMatrix` signature pqc_diagnostics.metrics'
    measurement functions expect via `.noisy_simulate`.
    """

    def __init__(self, backend: BackendV2, max_patch_qubits: int, basis_gates: list[str]):
        self.backend = backend
        self.max_patch_qubits = max_patch_qubits
        self.basis_gates = basis_gates
        self._noise_model = NoiseModel.from_backend(backend)
        self._patch_edges = [
            (u, v)
            for u, v in backend.coupling_map.get_edges()
            if u < max_patch_qubits and v < max_patch_qubits
        ]
        self._simulator = AerSimulator(method="density_matrix", noise_model=self._noise_model)

    def transpile_for_noise(self, circuit: QuantumCircuit, seed: int = 0) -> QuantumCircuit:
        """Transpile a (still-parameterized) circuit onto the fixed qubit patch.

        Transpiling before binding parameters means this only needs to happen
        once per (spec, role) - not once per random parameter draw or per
        seed - since routing/optimization doesn't depend on the bound values.
        A fixed seed_transpiler keeps the routing decision (and thus gate
        counts) reproducible across runs.
        """
        n = circuit.num_qubits
        if n > self.max_patch_qubits:
            raise ValueError(f"circuit has {n} qubits, exceeds the fixed {self.max_patch_qubits}-qubit patch")
        sub_edges = [(u, v) for u, v in self._patch_edges if u < n and v < n]
        return transpile(
            circuit,
            coupling_map=CouplingMap(sub_edges),
            basis_gates=self.basis_gates,
            optimization_level=3,
            seed_transpiler=seed,
        )

    def noisy_simulate(self, circuit: QuantumCircuit, params: np.ndarray) -> DensityMatrix:
        """Bind `params` onto an already-transpiled circuit and return its
        density matrix under the fixed patch's calibrated noise."""
        bound = circuit.assign_parameters(params)
        bound.save_density_matrix()
        result = self._simulator.run(bound).result()
        return DensityMatrix(result.data()["density_matrix"])
