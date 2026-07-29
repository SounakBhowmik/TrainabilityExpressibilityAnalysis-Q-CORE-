import numpy as np
import pytest

from qcore_expr_train.circuits import CircuitSpec
from qcore_expr_train.noise import MAX_PATCH_QUBITS, noisy_simulate, transpile_for_noise


@pytest.mark.parametrize("num_qubits", [4, 6, 8, 10])
def test_transpile_stays_at_circuit_width(num_qubits: int):
    # The whole point of noise.py is avoiding the padding-to-full-backend-width
    # blowup - confirm the transpiled circuit stays at the circuit's own qubit
    # count, not FakeSherbrooke's 127.
    spec = CircuitSpec(feature_map_family="zz", ansatz_family="real_amplitudes", num_qubits=num_qubits, depth=2, topology="linear")
    transpiled = transpile_for_noise(spec.feature_map())
    assert transpiled.num_qubits == num_qubits


def test_transpile_rejects_circuits_wider_than_the_patch():
    spec = CircuitSpec(feature_map_family="zz", ansatz_family="real_amplitudes", num_qubits=MAX_PATCH_QUBITS + 1, depth=1, topology="linear")
    with pytest.raises(ValueError):
        transpile_for_noise(spec.feature_map())


def test_topology_changes_two_qubit_gate_count():
    counts = {}
    for topology in ["linear", "circular", "full"]:
        spec = CircuitSpec(feature_map_family="zz", ansatz_family="real_amplitudes", num_qubits=6, depth=2, topology=topology)
        transpiled = transpile_for_noise(spec.feature_map())
        counts[topology] = transpiled.count_ops().get("ecr", 0)

    # A sparser topology needs less routing (fewer SWAPs) on the fixed patch's
    # coupling map - this is the mechanism behind the "noise pulls the ceiling
    # in, topology-dependent" claim.
    assert counts["linear"] < counts["circular"] < counts["full"]


def test_noisy_simulate_returns_valid_density_matrix():
    spec = CircuitSpec(feature_map_family="zz", ansatz_family="real_amplitudes", num_qubits=4, depth=1, topology="linear")
    transpiled = transpile_for_noise(spec.feature_map())
    params = np.random.default_rng(0).uniform(0, 2 * np.pi, transpiled.num_parameters)

    dm = noisy_simulate(transpiled, params)

    assert dm.dim == 2**4
    assert np.trace(dm.data).real == pytest.approx(1.0, abs=1e-6)
    # A valid density matrix is Hermitian.
    assert np.allclose(dm.data, dm.data.conj().T, atol=1e-8)
