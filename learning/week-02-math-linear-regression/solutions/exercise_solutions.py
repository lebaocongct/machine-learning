"""Reference functions for exercises/starter.py."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_value = _vector(a, "a")
    b_value = _vector(b, "b")
    if a_value.shape != b_value.shape:
        raise ValueError("a and b must have the same shape")
    denominator = np.linalg.norm(a_value) * np.linalg.norm(b_value)
    if denominator == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float((a_value @ b_value) / denominator)


def numerical_derivative(
    function: Callable[[float], float], x: float, epsilon: float = 1e-6
) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float((function(x + epsilon) - function(x - epsilon)) / (2 * epsilon))


def relative_gradient_error(
    analytical: np.ndarray, numerical: np.ndarray
) -> float:
    analytical_value = _vector(analytical, "analytical")
    numerical_value = _vector(numerical, "numerical")
    if analytical_value.shape != numerical_value.shape:
        raise ValueError("gradient shapes must match")
    numerator = np.linalg.norm(analytical_value - numerical_value)
    denominator = max(
        1.0,
        np.linalg.norm(analytical_value) + np.linalg.norm(numerical_value),
    )
    return float(numerator / denominator)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = _vector(y_true, "y_true")
    predicted = _vector(y_pred, "y_pred")
    if true.shape != predicted.shape:
        raise ValueError("target shapes must match")
    return float(np.sqrt(np.mean(np.square(predicted - true))))


def r2_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = _vector(y_true, "y_true")
    predicted = _vector(y_pred, "y_pred")
    if true.shape != predicted.shape:
        raise ValueError("target shapes must match")
    total = np.sum(np.square(true - true.mean()))
    if total == 0:
        raise ValueError("R2 is undefined for a constant target")
    residual = np.sum(np.square(true - predicted))
    return float(1.0 - residual / total)

