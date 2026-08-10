import pytest
from qiskit.circuit import QuantumCircuit

from pqc_diagnostics import registry


def _toy_feature_map(feature_dimension: int, reps: int, entanglement: str) -> QuantumCircuit:
    del entanglement
    qc = QuantumCircuit(feature_dimension)
    for _ in range(reps):
        for i in range(feature_dimension):
            qc.rz(0.0, i)
    return qc


def _toy_ansatz(num_qubits: int, reps: int, entanglement: str) -> QuantumCircuit:
    del entanglement
    qc = QuantumCircuit(num_qubits)
    for _ in range(reps):
        for i in range(num_qubits):
            qc.ry(0.0, i)
    return qc


def test_builtin_feature_maps_and_ansatze_are_preregistered():
    # circuits.py registers these at import time - importing pqc_diagnostics
    # (which pulls in circuits.py) must make them available with no other
    # setup, since qcore_expr_train's grid relies on this.
    import pqc_diagnostics.circuits  # noqa: F401 - triggers registration

    assert set(registry.registered_feature_maps()) >= {"zz", "local_z", "cx_ry"}
    assert set(registry.registered_ansatze()) >= {"rotation_only", "real_amplitudes"}


def test_register_and_get_feature_map_round_trip():
    registry.register_feature_map("test_toy_fm", _toy_feature_map, overwrite=True)
    assert "test_toy_fm" in registry.registered_feature_maps()
    assert registry.get_feature_map("test_toy_fm") is _toy_feature_map


def test_register_and_get_ansatz_round_trip():
    registry.register_ansatz("test_toy_ansatz", _toy_ansatz, overwrite=True)
    assert "test_toy_ansatz" in registry.registered_ansatze()
    assert registry.get_ansatz("test_toy_ansatz") is _toy_ansatz


def test_registering_a_duplicate_name_without_overwrite_raises():
    registry.register_feature_map("test_dup_fm", _toy_feature_map, overwrite=True)
    with pytest.raises(ValueError):
        registry.register_feature_map("test_dup_fm", _toy_feature_map, overwrite=False)


def test_getting_an_unregistered_name_raises_key_error():
    with pytest.raises(KeyError):
        registry.get_feature_map("definitely_not_registered")
    with pytest.raises(KeyError):
        registry.get_ansatz("definitely_not_registered")
