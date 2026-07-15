"""Starter implementation for the Week 2 NumPy regression challenge."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Return a new 2D array with a leading column of ones."""
    raise NotImplementedError("TODO: implement add_bias_column")


def predict_linear(X_bias: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return predictions for a bias-augmented design matrix."""
    raise NotImplementedError("TODO: implement predict_linear")


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean squared error as a Python float."""
    raise NotImplementedError("TODO: implement mean_squared_error")


def mse_gradient(X_bias: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return the analytical gradient of MSE with respect to theta."""
    raise NotImplementedError("TODO: implement mse_gradient")


def finite_difference_gradient(
    loss_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Estimate a gradient using central finite differences."""
    raise NotImplementedError("TODO: implement finite_difference_gradient")


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return column means and safe population standard deviations."""
    raise NotImplementedError("TODO: implement fit_standardizer")


def apply_standardizer(
    X: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply a previously fitted standardizer without mutating X."""
    raise NotImplementedError("TODO: implement apply_standardizer")


def gradient_descent(
    X_bias: np.ndarray,
    y: np.ndarray,
    initial_theta: np.ndarray,
    learning_rate: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run full-batch gradient descent and return theta and loss history."""
    raise NotImplementedError("TODO: implement gradient_descent")


class LinearRegressionGD:
    """Linear regression trained with standardized full-batch GD."""

    def __init__(self, learning_rate: float = 0.05, n_steps: int = 2_000) -> None:
        self.learning_rate = learning_rate
        self.n_steps = n_steps

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        raise NotImplementedError("TODO: implement fit")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: implement predict")

