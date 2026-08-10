"""Computes per-circuit PQC diagnostics: expressibility, entangling
capability, trainability, and kernel usefulness.

Each function takes a `simulate` callable (circuit, params) -> state, so the
same code path covers both noiseless and noisy simulation: the default is
exact, noiseless statevector simulation; passing a `simulate` that returns a
DensityMatrix from an already-noise-transpiled circuit (see
pqc_diagnostics.hardware.providers.NoiseProvider) covers the noisy condition
with no other code changes, since state_fidelity/partial_trace/purity/
expectation_value all accept DensityMatrix as well as Statevector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector, partial_trace, purity, state_fidelity
from sklearn.svm import SVC

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


def _fidelity_matrix(
    feature_map: QuantumCircuit,
    X1: np.ndarray,
    X2: np.ndarray | None = None,
    simulate: Simulate = _default_simulate,
) -> np.ndarray:
    """Pairwise state-fidelity matrix between encoded X1 and X2 (or X1 with
    itself). This *is* the quantum kernel matrix, computed directly by
    state overlap rather than a shot-based fidelity primitive - exact but
    O(|X1|*|X2|), which is why callers should keep sample sizes small.
    """
    states1 = [simulate(feature_map, x) for x in X1]
    states2 = states1 if X2 is None else [simulate(feature_map, x) for x in X2]

    K = np.empty((len(states1), len(states2)))
    for i, s1 in enumerate(states1):
        for j, s2 in enumerate(states2):
            K[i, j] = state_fidelity(s1, s2)
    return K


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Kernel-target alignment: cosine similarity, in Frobenius inner-product
    space, between the kernel matrix and the ideal same-class-similarity
    matrix yy^T. Higher means the kernel's notion of similarity agrees more
    with the class labels.
    """
    labels = np.where(y == y[0], 1, -1).astype(float) if len(np.unique(y)) == 2 else y.astype(float)
    Y = np.outer(labels, labels)
    return float(np.sum(K * Y) / (np.linalg.norm(K) * np.linalg.norm(Y)))


@dataclass(frozen=True)
class KernelUsefulnessResult:
    kta: float
    svm_accuracy: float


def kernel_usefulness(
    feature_map: QuantumCircuit,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    simulate: Simulate = _default_simulate,
) -> KernelUsefulnessResult:
    """KTA on the training kernel plus held-out SVM classification accuracy -
    the two "is this kernel actually useful for the task" measurements. Uses
    the same pluggable `simulate` signature as the other three metrics, so it
    covers the noisy condition the same way."""
    K_train = _fidelity_matrix(feature_map, X_train, simulate=simulate)
    kta = kernel_target_alignment(K_train, y_train)

    svc = SVC(kernel="precomputed").fit(K_train, y_train)
    K_test = _fidelity_matrix(feature_map, X_test, X_train, simulate=simulate)
    accuracy = float(svc.score(K_test, y_test))

    return KernelUsefulnessResult(kta=kta, svm_accuracy=accuracy)
