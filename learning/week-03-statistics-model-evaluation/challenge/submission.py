"""Starter implementation for the Week 3 leakage-safe model-selection challenge."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


def population_summary(values: np.ndarray, ddof: int = 0) -> dict[str, float]:
    """Return mean, variance and standard deviation for finite numeric values."""
    raise NotImplementedError("TODO: implement population_summary")


def conditional_probability(event: np.ndarray, given: np.ndarray) -> float:
    """Return P(event | given) for aligned boolean arrays."""
    raise NotImplementedError("TODO: implement conditional_probability")


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Return estimate and a percentile bootstrap confidence interval."""
    raise NotImplementedError("TODO: implement bootstrap_ci")


def three_way_split_indices(
    n_samples: int,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic, disjoint and exhaustive train/validation/test indices."""
    raise NotImplementedError("TODO: implement three_way_split_indices")


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return RMSE as a Python float."""
    raise NotImplementedError("TODO: implement root_mean_squared_error")


def build_polynomial_model(degree: int):
    """Return an unfitted scikit-learn polynomial-regression Pipeline."""
    raise NotImplementedError("TODO: implement build_polynomial_model")


def evaluate_degrees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    degrees: Iterable[int],
) -> pd.DataFrame:
    """Compare degrees using train and validation only."""
    raise NotImplementedError("TODO: implement evaluate_degrees")


def select_best_degree(results: pd.DataFrame) -> int:
    """Select the smallest degree among minimum validation RMSE ties."""
    raise NotImplementedError("TODO: implement select_best_degree")


def manual_learning_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    degree: int,
    train_sizes: Iterable[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit nested training subsets and return train/validation RMSE by size."""
    raise NotImplementedError("TODO: implement manual_learning_curve")


class LeakageSafePolynomialWorkflow:
    """Three-way split, validation-only selection and one final test evaluation."""

    def __init__(
        self,
        degrees: Iterable[int],
        *,
        train_fraction: float = 0.60,
        validation_fraction: float = 0.20,
        test_fraction: float = 0.20,
        random_state: int = 42,
    ) -> None:
        self.degrees = tuple(degrees)
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LeakageSafePolynomialWorkflow":
        raise NotImplementedError("TODO: implement fit")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: implement predict")
