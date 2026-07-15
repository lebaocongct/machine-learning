"""Reference implementation of linear regression with NumPy gradient descent."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    value = np.asarray(X, dtype=float)
    if value.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("X must not be empty")
    if not np.isfinite(value).all():
        raise ValueError("X must contain only finite values")
    return value


def _as_1d_float(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def add_bias_column(X: np.ndarray) -> np.ndarray:
    X_value = _as_2d_float(X)
    return np.column_stack([np.ones(X_value.shape[0]), X_value])


def predict_linear(X_bias: np.ndarray, theta: np.ndarray) -> np.ndarray:
    X_value = _as_2d_float(X_bias)
    theta_value = _as_1d_float(theta, "theta")
    if X_value.shape[1] != theta_value.size:
        raise ValueError("X_bias columns must match theta size")
    return X_value @ theta_value


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = _as_1d_float(y_true, "y_true")
    predicted = _as_1d_float(y_pred, "y_pred")
    if true.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.square(predicted - true)))


def mse_gradient(X_bias: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    X_value = _as_2d_float(X_bias)
    y_value = _as_1d_float(y, "y")
    theta_value = _as_1d_float(theta, "theta")
    if X_value.shape[0] != y_value.size:
        raise ValueError("X_bias rows must match y size")
    if X_value.shape[1] != theta_value.size:
        raise ValueError("X_bias columns must match theta size")
    residual = X_value @ theta_value - y_value
    return (2.0 / X_value.shape[0]) * (X_value.T @ residual)


def finite_difference_gradient(
    loss_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    theta_value = _as_1d_float(theta, "theta").copy()
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive")
    gradient = np.empty_like(theta_value)
    for index in range(theta_value.size):
        plus = theta_value.copy()
        minus = theta_value.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (float(loss_fn(plus)) - float(loss_fn(minus))) / (
            2.0 * epsilon
        )
    return gradient


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_value = _as_2d_float(X)
    mean = X_value.mean(axis=0)
    scale = X_value.std(axis=0, ddof=0)
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    return mean, safe_scale


def apply_standardizer(
    X: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    X_value = _as_2d_float(X)
    mean_value = _as_1d_float(mean, "mean")
    scale_value = _as_1d_float(scale, "scale")
    if X_value.shape[1] != mean_value.size or mean_value.shape != scale_value.shape:
        raise ValueError("standardizer parameters do not match X columns")
    if np.any(scale_value <= 0):
        raise ValueError("scale must be positive")
    return (X_value - mean_value) / scale_value


def gradient_descent(
    X_bias: np.ndarray,
    y: np.ndarray,
    initial_theta: np.ndarray,
    learning_rate: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_value = _as_2d_float(X_bias)
    y_value = _as_1d_float(y, "y")
    theta = _as_1d_float(initial_theta, "initial_theta").copy()
    if X_value.shape[0] != y_value.size or X_value.shape[1] != theta.size:
        raise ValueError("incompatible X_bias, y and theta shapes")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")

    history = np.empty(n_steps + 1, dtype=float)
    for step in range(n_steps + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            residual = X_value @ theta - y_value
            loss = float(np.mean(np.square(residual)))
        if not np.isfinite(loss):
            raise FloatingPointError(
                "loss became non-finite; reduce the learning rate or scale features"
            )
        history[step] = loss
        if step < n_steps:
            theta -= learning_rate * mse_gradient(X_value, y_value, theta)
    return theta, history


class LinearRegressionGD:
    def __init__(self, learning_rate: float = 0.05, n_steps: int = 2_000) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        self.learning_rate = learning_rate
        self.n_steps = n_steps

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X_value = _as_2d_float(X)
        y_value = _as_1d_float(y, "y")
        if X_value.shape[0] != y_value.size:
            raise ValueError("X rows must match y size")

        self.n_features_in_ = X_value.shape[1]
        self.mean_, self.scale_ = fit_standardizer(X_value)
        X_scaled = apply_standardizer(X_value, self.mean_, self.scale_)
        X_bias = add_bias_column(X_scaled)
        initial_theta = np.zeros(X_bias.shape[1], dtype=float)
        self.theta_, self.loss_history_ = gradient_descent(
            X_bias,
            y_value,
            initial_theta,
            self.learning_rate,
            self.n_steps,
        )

        self.coef_ = self.theta_[1:] / self.scale_
        self.intercept_ = float(
            self.theta_[0] - np.dot(self.coef_, self.mean_)
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before predict")
        X_value = _as_2d_float(X)
        if X_value.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than fit data")
        return self.intercept_ + X_value @ self.coef_
