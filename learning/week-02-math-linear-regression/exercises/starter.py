"""Starter functions for the Week 2 independent exercises."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity for two non-zero 1D vectors."""
    raise NotImplementedError


def numerical_derivative(
    function: Callable[[float], float], x: float, epsilon: float = 1e-6
) -> float:
    """Estimate a scalar derivative using central differences."""
    raise NotImplementedError


def relative_gradient_error(
    analytical: np.ndarray, numerical: np.ndarray
) -> float:
    """Return the relative L2 error used in this week's gradient check."""
    raise NotImplementedError


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return RMSE as a Python float."""
    raise NotImplementedError


def r2_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of determination without Scikit-Learn."""
    raise NotImplementedError

