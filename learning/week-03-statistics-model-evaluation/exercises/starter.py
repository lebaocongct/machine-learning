"""Optional starter functions for the independent Week 3 exercises."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def descriptive_summary(values: np.ndarray) -> dict[str, float]:
    """Return mean, median, variance(ddof=1), std(ddof=1), q25 and q75."""
    raise NotImplementedError


def probability_table(first: np.ndarray, second: np.ndarray) -> pd.DataFrame:
    """Return a normalized 2x2 joint probability table for boolean events."""
    raise NotImplementedError


def simulate_sample_means(
    population: np.ndarray,
    *,
    sample_size: int,
    n_trials: int,
    random_state: int,
) -> np.ndarray:
    """Return repeated sample means from a finite population."""
    raise NotImplementedError


def percentile_bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int,
    confidence_level: float,
    random_state: int,
) -> tuple[float, float, float]:
    """Return estimate, lower and upper percentile bootstrap bounds."""
    raise NotImplementedError


def assert_split_invariants(
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    n_samples: int,
) -> None:
    """Raise AssertionError unless splits are disjoint and exhaustive."""
    raise NotImplementedError


def diagnose_learning_curve(
    train_rmse: np.ndarray, validation_rmse: np.ndarray
) -> str:
    """Return one of: high_bias, high_variance, converged, insufficient_evidence."""
    raise NotImplementedError
