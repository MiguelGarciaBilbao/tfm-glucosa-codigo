"""Baseline 1: ridge regression on the flattened history window.
"""

from __future__ import annotations

import numpy as np


class RidgeForecaster:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1).astype(np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeForecaster":
        Z = self._flatten(X)
        y = np.asarray(y, dtype=np.float64)
        z_mean, y_mean = Z.mean(axis=0), y.mean(axis=0)
        Zc, yc = Z - z_mean, y - y_mean
        n_features = Zc.shape[1]
        gram = Zc.T @ Zc + self.alpha * np.eye(n_features)
        self.coef_ = np.linalg.solve(gram, Zc.T @ yc)
        self.intercept_ = y_mean - z_mean @ self.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeForecaster must be fitted before predicting")
        return (self._flatten(X) @ self.coef_ + self.intercept_).astype(np.float32)
