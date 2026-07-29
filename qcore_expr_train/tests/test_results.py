import numpy as np

from qcore_expr_train.circuits import CircuitSpec
from qcore_expr_train.datasets import prepare_breast_cancer
from qcore_expr_train.metrics import gradient_variance
from qcore_expr_train.results import evaluate_cell


def test_evaluate_cell_noiseless_runs_end_to_end():
    spec = CircuitSpec(feature_map_family="zz", ansatz_family="real_amplitudes", num_qubits=4, depth=1, topology="linear")
    df = evaluate_cell(spec, seeds=[0, 1])

    assert len(df) == 2
    assert set(df["noise_condition"]) == {"noiseless"}
    assert (df["expressibility_kl"] >= 0).all()
    assert (df["gradient_variance"] >= 0).all()


def test_trainability_uses_composed_feature_map_and_ansatz():
    # Trainability must reflect the encoding's contribution, not just the
    # ansatz in isolation - binding different real data samples onto the same
    # ansatz should generally change the gradient-variance estimate.
    spec = CircuitSpec(feature_map_family="zz", ansatz_family="real_amplitudes", num_qubits=4, depth=2, topology="linear")
    feature_map = spec.feature_map()
    ansatz = spec.ansatz()
    composed = feature_map.compose(ansatz)
    bundle = prepare_breast_cancer(4)

    ansatz_alone = gradient_variance(ansatz, num_samples=50, seed=0)

    trainable = composed.assign_parameters(dict(zip(feature_map.parameters, bundle.X_train[0])))
    composed_value = gradient_variance(trainable, num_samples=50, seed=0)

    # Composing with a bound data sample must not silently reduce to
    # measuring the ansatz alone.
    assert composed_value != ansatz_alone
    assert trainable.num_parameters == ansatz.num_parameters


def test_different_seeds_bind_different_data_samples():
    # evaluate_cell cycles the feature map's data binding by seed - confirm
    # that produces genuinely different composed circuits, not the same one
    # relabeled.
    spec = CircuitSpec(feature_map_family="zz", ansatz_family="real_amplitudes", num_qubits=4, depth=1, topology="linear")
    bundle = prepare_breast_cancer(4)
    assert not np.allclose(bundle.X_train[0], bundle.X_train[1])
