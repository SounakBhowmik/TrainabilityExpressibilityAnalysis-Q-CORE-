import numpy as np
import pytest
from qiskit.circuit.library import zz_feature_map
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from pqc_diagnostics.hardware.providers import NoiseProvider
from pqc_diagnostics.hardware.registry import get_provider, register_provider

# NoiseProvider itself is backend-agnostic (takes any BackendV2) - FakeSherbrooke
# is just a convenient real, calibrated backend to exercise it against here.
_PROVIDER = NoiseProvider(backend=FakeSherbrooke(), max_patch_qubits=5, basis_gates=["sx", "rz", "x", "ecr"])


def test_transpile_stays_at_circuit_width():
    fm = zz_feature_map(feature_dimension=4, reps=1)
    transpiled = _PROVIDER.transpile_for_noise(fm)
    assert transpiled.num_qubits == 4


def test_transpile_rejects_circuits_wider_than_the_patch():
    fm = zz_feature_map(feature_dimension=6, reps=1)
    with pytest.raises(ValueError):
        _PROVIDER.transpile_for_noise(fm)


def test_noisy_simulate_returns_valid_density_matrix():
    fm = zz_feature_map(feature_dimension=3, reps=1)
    transpiled = _PROVIDER.transpile_for_noise(fm)
    params = np.random.default_rng(0).uniform(0, 2 * np.pi, transpiled.num_parameters)

    dm = _PROVIDER.noisy_simulate(transpiled, params)

    assert dm.dim == 2**3
    assert np.trace(dm.data).real == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(dm.data, dm.data.conj().T, atol=1e-8)


def test_provider_registry_round_trip():
    register_provider("test_sherbrooke_5q", _PROVIDER, overwrite=True)
    assert get_provider("test_sherbrooke_5q") is _PROVIDER
