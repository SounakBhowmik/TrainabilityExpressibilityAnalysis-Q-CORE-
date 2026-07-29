"""Computes the three per-grid-cell measurements from the experiment plan:
expressibility, entangling capability, and trainability.

Each function takes a `simulate` callable (circuit, params) -> state, so the
same code path covers both conditions: the default is exact, noiseless
statevector simulation (cheap, used for the pilot and the "noiseless" full-grid
condition); the full-grid "noisy" condition instead passes
`noise.noisy_simulate`, which returns a DensityMatrix from an already-noise-
transpiled circuit. state_fidelity/partial_trace/purity/expectation_value all
accept DensityMatrix as well as Statevector, so no other code changes are
needed to support noise.

Kernel usefulness (KTA + SVM accuracy) was dropped - it wasn't feeding any
poster plot, and in the noisy condition its fidelity-matrix Aer calls (one
per data sample) were roughly half the per-seed cost. Deferred, not deleted
from history - see git log if it's needed again.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector, partial_trace, purity, state_fidelity

State = Union[Statevector, DensityMatrix]
Simulate = Callable[[QuantumCircuit, np.ndarray], State]


def _default_simulate(circuit: QuantumCircuit, params: np.ndarray) -> Statevector:
    return Statevector(circuit.assign_parameters(params))


def _random_params(circuit: QuantumCircuit, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0, 2 * np.pi, size=circuit.num_parameters)


def expressibility_kl(
    circuit: QuantumCircuit,
    num_samples: int = 500,
    num_bins: int = 50,
    seed: int = 0,
    simulate: Simulate = _default_simulate,
) -> float:
    """KL divergence between the circuit's sampled fidelity distribution and
    the analytic Haar-random fidelity distribution (Sim et al. 2019,
    "Expressibility and entangling capability of parameterized quantum
    circuits"). Lower means more expressible - closer to covering the full
    Hilbert space the way a Haar-random circuit would.
    """
    rng = np.random.default_rng(seed)
    n = circuit.num_qubits
    fidelities = np.empty(num_samples)
    for i in range(num_samples):
        s1 = simulate(circuit, _random_params(circuit, rng))
        s2 = simulate(circuit, _random_params(circuit, rng))
        fidelities[i] = state_fidelity(s1, s2)

    hist, edges = np.histogram(fidelities, bins=num_bins, range=(0, 1))
    p_circuit = hist / hist.sum()

    dim = 2**n
    centers = (edges[:-1] + edges[1:]) / 2
    p_haar = (dim - 1) * (1 - centers) ** (dim - 2)
    p_haar = p_haar / p_haar.sum()

    nonzero = p_circuit > 0
    return float(np.sum(p_circuit[nonzero] * np.log(p_circuit[nonzero] / p_haar[nonzero])))


def meyer_wallach(
    circuit: QuantumCircuit,
    num_samples: int = 200,
    seed: int = 0,
    simulate: Simulate = _default_simulate,
) -> float:
    """Meyer-Wallach entangling capability, averaged over random parameter
    draws. Q = 2*(1 - mean_k Tr(rho_k^2)) ranges from 0 (product state, no
    entanglement) to 1 (maximally entangled).
    """
    rng = np.random.default_rng(seed)
    n = circuit.num_qubits
    q_values = np.empty(num_samples)
    for i in range(num_samples):
        state = simulate(circuit, _random_params(circuit, rng))
        purities = [purity(partial_trace(state, [q for q in range(n) if q != k])).real for k in range(n)]
        q_values[i] = 2 * (1 - np.mean(purities))
    return float(np.mean(q_values))


def gradient_variance(
    circuit: QuantumCircuit,
    num_samples: int = 200,
    seed: int = 0,
    simulate: Simulate = _default_simulate,
) -> float:
    """Variance of d<Z_0>/dtheta_0 across random parameter draws, via the
    parameter-shift rule (McClean et al. 2018 barren-plateau protocol).
    Variance collapsing toward zero as circuits get wider/deeper is the
    trainability side of the expressibility-trainability tradeoff.
    """
    rng = np.random.default_rng(seed)
    n = circuit.num_qubits
    observable = SparsePauliOp("I" * (n - 1) + "Z")
    shift = np.pi / 2
    grads = np.empty(num_samples)
    for i in range(num_samples):
        theta = _random_params(circuit, rng)
        theta_plus, theta_minus = theta.copy(), theta.copy()
        theta_plus[0] += shift
        theta_minus[0] -= shift
        exp_plus = simulate(circuit, theta_plus).expectation_value(observable).real
        exp_minus = simulate(circuit, theta_minus).expectation_value(observable).real
        grads[i] = (exp_plus - exp_minus) / 2
    return float(np.var(grads))
