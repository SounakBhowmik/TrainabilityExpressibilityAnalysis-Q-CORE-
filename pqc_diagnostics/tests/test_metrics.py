import numpy as np
from qiskit.circuit.library import zz_feature_map

from pqc_diagnostics.metrics import (
    KernelUsefulnessResult,
    expressibility_kl,
    gradient_variance,
    kernel_usefulness,
    meyer_wallach,
)


def test_expressibility_kl_is_nonnegative():
    fm = zz_feature_map(feature_dimension=3, reps=1)
    assert expressibility_kl(fm, num_samples=30, seed=0) >= 0


def test_meyer_wallach_is_in_valid_range():
    fm = zz_feature_map(feature_dimension=3, reps=2)
    q = meyer_wallach(fm, num_samples=30, seed=0)
    assert 0 <= q <= 1 + 1e-9


def test_gradient_variance_is_nonnegative():
    fm = zz_feature_map(feature_dimension=3, reps=1)
    assert gradient_variance(fm, num_samples=30, seed=0) >= 0


def test_kernel_usefulness_returns_kta_and_svm_accuracy_in_valid_ranges():
    # New metric (never existed in this repo's history) - basic sanity: KTA
    # is a cosine similarity (bounded [-1, 1]) and SVM accuracy is a
    # fraction (bounded [0, 1]).
    fm = zz_feature_map(feature_dimension=2, reps=1)
    rng = np.random.default_rng(0)
    X_train = rng.uniform(0, np.pi, size=(6, 2))
    y_train = np.array([0, 1, 0, 1, 0, 1])
    X_test = rng.uniform(0, np.pi, size=(4, 2))
    y_test = np.array([0, 1, 0, 1])

    result = kernel_usefulness(fm, X_train, y_train, X_test, y_test)

    assert isinstance(result, KernelUsefulnessResult)
    assert -1 - 1e-9 <= result.kta <= 1 + 1e-9
    assert 0 <= result.svm_accuracy <= 1
