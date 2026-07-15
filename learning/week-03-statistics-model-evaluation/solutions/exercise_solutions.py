"""Compact reference functions and checks for Week 3 exercises."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def descriptive_summary(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        raise ValueError("values must be a finite 1D array with at least two items")
    q25, q75 = np.quantile(data, [0.25, 0.75])
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "sample_variance": float(np.var(data, ddof=1)),
        "sample_standard_deviation": float(np.std(data, ddof=1)),
        "q25": float(q25),
        "q75": float(q75),
    }


def probability_table(first: np.ndarray, second: np.ndarray) -> pd.DataFrame:
    a = np.asarray(first)
    b = np.asarray(second)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("events must have the same non-empty 1D shape")
    return pd.crosstab(
        pd.Series(a.astype(bool), name="first"),
        pd.Series(b.astype(bool), name="second"),
        normalize="all",
        dropna=False,
    ).reindex(index=[False, True], columns=[False, True], fill_value=0.0)


def simulate_sample_means(
    population: np.ndarray,
    *,
    sample_size: int,
    n_trials: int,
    random_state: int,
) -> np.ndarray:
    data = np.asarray(population, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("population must be a finite non-empty 1D array")
    if sample_size < 1 or n_trials < 1:
        raise ValueError("sample_size and n_trials must be positive")
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, data.size, size=(n_trials, sample_size))
    return data[indices].mean(axis=1)


def percentile_bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int,
    confidence_level: float,
    random_state: int,
) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("values must be finite, non-empty and 1D")
    rng = np.random.default_rng(random_state)
    distribution = np.array(
        [statistic(data[rng.integers(0, data.size, data.size)]) for _ in range(n_resamples)],
        dtype=float,
    )
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(distribution, [alpha, 1 - alpha])
    return float(statistic(data)), float(lower), float(upper)


def assert_split_invariants(
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    n_samples: int,
) -> None:
    parts = [np.asarray(part) for part in (train_indices, validation_indices, test_indices)]
    assert all(part.ndim == 1 for part in parts)
    combined = np.concatenate(parts)
    assert len(combined) == n_samples
    assert len(np.unique(combined)) == n_samples
    np.testing.assert_array_equal(np.sort(combined), np.arange(n_samples))


def diagnose_learning_curve(
    train_rmse: np.ndarray, validation_rmse: np.ndarray
) -> str:
    train = np.asarray(train_rmse, dtype=float)
    validation = np.asarray(validation_rmse, dtype=float)
    if train.ndim != 1 or validation.shape != train.shape or train.size < 3:
        return "insufficient_evidence"
    final_train = train[-1]
    final_validation = validation[-1]
    gap = final_validation - final_train
    reference = max(1e-12, abs(final_validation))
    if gap / reference > 0.25:
        return "high_variance"
    if final_validation > np.min(validation) * 1.25:
        return "high_bias"
    return "converged"


LEAKAGE_CASES = {
    "standardize_before_split": "preprocessing leakage",
    "same_customer_across_splits": "duplicate/entity leakage",
    "post_outcome_feature": "target leakage",
    "choose_by_test_rmse": "test contamination",
    "fit_train_transform_holdouts": "valid",
    "random_split_drift_series": "temporal leakage",
    "select_threshold_on_validation": "valid",
    "choose_best_test_seed": "test contamination",
}
