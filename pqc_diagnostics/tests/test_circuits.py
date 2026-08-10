from qiskit.circuit import ParameterVector, QuantumCircuit

from pqc_diagnostics.circuits import CircuitSpec, build_grid
from pqc_diagnostics.registry import register_ansatz, register_feature_map


def _custom_feature_map(feature_dimension: int, reps: int, entanglement: str) -> QuantumCircuit:
    del entanglement
    x = ParameterVector("y", feature_dimension)
    qc = QuantumCircuit(feature_dimension)
    for _ in range(reps):
        for i in range(feature_dimension):
            qc.rx(x[i], i)
    return qc


def _custom_ansatz(num_qubits: int, reps: int, entanglement: str) -> QuantumCircuit:
    del entanglement
    qc = QuantumCircuit(num_qubits)
    for _ in range(reps):
        for i in range(num_qubits):
            qc.ry(0.3, i)
    return qc


def test_a_users_own_registered_feature_map_and_ansatz_work_end_to_end():
    # This is the whole point of the registry: a user brings their own PQC
    # without editing this library's source.
    register_feature_map("custom_rx", _custom_feature_map, overwrite=True)
    register_ansatz("custom_ry", _custom_ansatz, overwrite=True)

    spec = CircuitSpec(
        feature_map_family="custom_rx", ansatz_family="custom_ry", num_qubits=3, depth=2, topology="linear"
    )
    feature_map = spec.feature_map()
    ansatz = spec.ansatz()
    composed = spec.composed_circuit()

    assert feature_map.num_qubits == 3
    assert ansatz.num_qubits == 3
    assert composed.num_qubits == 3
    assert composed.num_parameters == feature_map.num_parameters + ansatz.num_parameters


def test_build_grid_defaults_to_currently_registered_families():
    register_feature_map("test_grid_default_fm", _custom_feature_map, overwrite=True)
    grid = build_grid(qubits=[3], depths=[1], topologies=["linear"], ansatz_families=["rotation_only"])
    families = {s.feature_map_family for s in grid}
    assert "test_grid_default_fm" in families


def test_build_grid_with_explicit_families_ignores_other_registered_ones():
    register_feature_map("test_grid_explicit_fm", _custom_feature_map, overwrite=True)
    grid = build_grid(
        qubits=[3], depths=[1], topologies=["linear"],
        feature_map_families=["zz"], ansatz_families=["rotation_only"],
    )
    assert {s.feature_map_family for s in grid} == {"zz"}
