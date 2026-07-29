import pytest

from qcore_expr_train.circuits import FULL_GRID, PILOT_GRID, CircuitSpec, build_grid


def test_pilot_grid_size():
    # 3 feature maps x 2 ansatze x 2 qubit counts x 2 depths x 2 topologies
    assert len(PILOT_GRID) == 3 * 2 * 2 * 2 * 2


@pytest.mark.parametrize("spec", PILOT_GRID)
def test_feature_map_and_ansatz_share_width_and_depth(spec: CircuitSpec):
    feature_map = spec.feature_map()
    ansatz = spec.ansatz()

    assert feature_map.num_qubits == spec.num_qubits
    assert ansatz.num_qubits == spec.num_qubits
    # Feature maps re-encode the same data parameters on every repetition
    # (that's what makes `spec.depth` a depth/reps knob rather than a
    # trainable-parameter-count knob), so the parameter count is always one
    # per qubit regardless of depth.
    assert feature_map.num_parameters == spec.num_qubits
    assert ansatz.num_parameters > 0


def test_rotation_only_ansatz_has_no_entangling_gates():
    spec = CircuitSpec(feature_map_family="zz", ansatz_family="rotation_only", num_qubits=4, depth=2, topology="linear")
    ansatz = spec.ansatz()

    two_qubit_gates = [instr for instr in ansatz.data if instr.operation.num_qubits > 1]
    assert two_qubit_gates == []


def test_real_amplitudes_ansatz_entangles():
    spec = CircuitSpec(feature_map_family="zz", ansatz_family="real_amplitudes", num_qubits=4, depth=2, topology="linear")
    ansatz = spec.ansatz()

    two_qubit_gates = [instr for instr in ansatz.data if instr.operation.num_qubits > 1]
    assert len(two_qubit_gates) > 0


def test_build_grid_is_cartesian_product():
    grid = build_grid(
        qubits=[3],
        depths=[1],
        topologies=["linear"],
        feature_map_families=["zz", "cx_ry"],
        ansatz_families=["rotation_only", "real_amplitudes"],
    )
    # 2 feature maps x 2 ansatze, crossed - not paired 1:1
    assert len(grid) == 4
    assert {(s.feature_map_family, s.ansatz_family) for s in grid} == {
        ("zz", "rotation_only"),
        ("zz", "real_amplitudes"),
        ("cx_ry", "rotation_only"),
        ("cx_ry", "real_amplitudes"),
    }


def test_cx_ry_feature_map_entangles_and_reuses_parameters():
    spec = CircuitSpec(feature_map_family="cx_ry", ansatz_family="real_amplitudes", num_qubits=4, depth=3, topology="linear")
    feature_map = spec.feature_map()

    # Same data parameters reused every repetition (like zz/pauli), not fresh
    # parameters per rep - otherwise it wouldn't behave like a feature map.
    assert feature_map.num_parameters == 4
    two_qubit_gates = [instr for instr in feature_map.data if instr.operation.num_qubits > 1]
    assert len(two_qubit_gates) > 0


def test_cx_ry_topology_changes_entangling_gate_count():
    counts = {}
    for topology in ["linear", "circular", "full"]:
        spec = CircuitSpec(feature_map_family="cx_ry", ansatz_family="rotation_only", num_qubits=4, depth=1, topology=topology)
        fm = spec.feature_map()
        counts[topology] = len([i for i in fm.data if i.operation.num_qubits > 1])

    # linear < circular < full, matching qiskit's own entanglement semantics
    assert counts["linear"] < counts["circular"] < counts["full"]


@pytest.mark.parametrize("spec", PILOT_GRID)
def test_composed_circuit_has_both_feature_map_and_ansatz_parameters(spec: CircuitSpec):
    composed = spec.composed_circuit()
    feature_map = spec.feature_map()
    ansatz = spec.ansatz()

    assert composed.num_qubits == spec.num_qubits
    assert composed.num_parameters == feature_map.num_parameters + ansatz.num_parameters


def test_the_two_ansatze_are_structurally_distinct():
    # The whole point of having exactly 2 ansatze is that they're genuinely
    # different circuits, so crossing them with the 3 feature maps produces
    # 2 distinct trainability curves per feature map, not 1.
    spec_a = CircuitSpec(feature_map_family="zz", ansatz_family="rotation_only", num_qubits=4, depth=2, topology="linear")
    spec_b = CircuitSpec(feature_map_family="zz", ansatz_family="real_amplitudes", num_qubits=4, depth=2, topology="linear")

    assert spec_a.ansatz().count_ops() != spec_b.ansatz().count_ops()


def test_feature_map_choice_is_independent_of_ansatz_choice():
    # Every feature map must be pairable with every ansatz - the crossed
    # design this project now relies on, replacing the earlier 1:1
    # family-to-ansatz pairing.
    for feature_map_family in ["zz", "local_z", "cx_ry"]:
        for ansatz_family in ["rotation_only", "real_amplitudes"]:
            spec = CircuitSpec(
                feature_map_family=feature_map_family,
                ansatz_family=ansatz_family,
                num_qubits=4,
                depth=1,
                topology="linear",
            )
            assert spec.feature_map().num_qubits == 4
            assert spec.ansatz().num_qubits == 4


def test_full_grid_size_after_topology_cut():
    # 3 qubits x 4 depths x (5 fm/ansatz pairs x 3 topologies + 1 pair x 1
    # topology) - local_z + rotation_only is the one pair with no entangling
    # gate anywhere, so it's fixed to a single topology instead of crossed
    # with all 3 (see circuits.FULL_GRID's comment for why that's safe).
    assert len(FULL_GRID) == 3 * 4 * (5 * 3 + 1 * 1)


def test_local_z_rotation_only_is_fixed_to_one_topology():
    cells = [s for s in FULL_GRID if s.feature_map_family == "local_z" and s.ansatz_family == "rotation_only"]
    assert cells  # sanity: the pair is still present in the grid
    assert {s.topology for s in cells} == {"linear"}


def test_every_other_pair_keeps_all_three_topologies():
    other_pairs = {
        (s.feature_map_family, s.ansatz_family)
        for s in FULL_GRID
        if not (s.feature_map_family == "local_z" and s.ansatz_family == "rotation_only")
    }
    for feature_map_family, ansatz_family in other_pairs:
        cells = [
            s
            for s in FULL_GRID
            if s.feature_map_family == feature_map_family and s.ansatz_family == ansatz_family
        ]
        assert {s.topology for s in cells} == {"linear", "circular", "full"}
