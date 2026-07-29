"""Prepares Breast Cancer Wisconsin for angle-encoded quantum circuits.

PCA reduces the dataset's 30 features down to one per qubit, since every
circuit family in circuits.py encodes exactly one feature per qubit. All
transforms (scaler, PCA, angle scaler) are fit on the training split only,
so no test-set information leaks into them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    num_qubits: int
    explained_variance_ratio: np.ndarray  # how much of the original 30 features PCA retained


def prepare_breast_cancer(
    num_qubits: int,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> DataBundle:
    """Standardize, PCA-reduce to `num_qubits` components, then angle-encode.

    Two StandardScaler passes are used deliberately: the first puts the 30
    raw features (very different units/scales) on comparable footing before
    PCA; the second normalizes the resulting PCA components (whose scale
    varies with explained variance) so they land in a consistent range
    before the pi/2 angle conversion.
    """
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    pre_scaler = StandardScaler().fit(X_train)
    X_train, X_test = pre_scaler.transform(X_train), pre_scaler.transform(X_test)

    pca = PCA(n_components=num_qubits, random_state=random_seed).fit(X_train)
    X_train, X_test = pca.transform(X_train), pca.transform(X_test)

    angle_scaler = StandardScaler().fit(X_train)
    X_train = angle_scaler.transform(X_train) * np.pi / 2
    X_test = angle_scaler.transform(X_test) * np.pi / 2

    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_qubits=num_qubits,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
