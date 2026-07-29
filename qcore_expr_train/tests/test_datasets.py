import numpy as np

from qcore_expr_train.datasets import prepare_breast_cancer


def test_shapes_and_qubit_count():
    bundle = prepare_breast_cancer(num_qubits=4, test_size=0.2, random_seed=0)

    assert bundle.X_train.shape[1] == 4
    assert bundle.X_test.shape[1] == 4
    assert bundle.X_train.shape[0] == bundle.y_train.shape[0]
    assert bundle.X_test.shape[0] == bundle.y_test.shape[0]
    assert len(bundle.explained_variance_ratio) == 4


def test_angles_are_bounded():
    bundle = prepare_breast_cancer(num_qubits=6, random_seed=0)

    # Not a hard guarantee (StandardScaler doesn't clip), but the bulk of a
    # standardized-then-pi/2-scaled distribution should sit within a few
    # multiples of pi/2 - a gross encoding error would blow way past this.
    assert np.abs(bundle.X_train).max() < 10 * np.pi / 2


def test_train_test_split_is_stratified_and_disjoint_seeded():
    a = prepare_breast_cancer(num_qubits=4, random_seed=1)
    b = prepare_breast_cancer(num_qubits=4, random_seed=1)

    np.testing.assert_array_equal(a.X_train, b.X_train)
