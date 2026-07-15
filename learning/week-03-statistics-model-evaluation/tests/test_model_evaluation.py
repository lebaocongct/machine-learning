"""Public tests for the Week 3 statistics/model-selection challenge."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


IMPL = importlib.import_module(os.getenv("WEEK3_IMPL", "challenge.submission"))


def test_population_summary_matches_numpy() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0])
    result = IMPL.population_summary(values)
    assert result["mean"] == pytest.approx(np.mean(values))
    assert result["variance"] == pytest.approx(np.var(values, ddof=0))
    assert result["standard_deviation"] == pytest.approx(np.std(values, ddof=0))
    sample = IMPL.population_summary(values, ddof=1)
    assert sample["variance"] == pytest.approx(np.var(values, ddof=1))


def test_population_summary_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        IMPL.population_summary(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError):
        IMPL.population_summary(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        IMPL.population_summary(np.array([1.0]), ddof=1)


def test_conditional_probability() -> None:
    premium = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    converted = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
    assert IMPL.conditional_probability(converted, premium) == pytest.approx(2 / 3)
    assert IMPL.conditional_probability(premium, converted) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="undefined"):
        IMPL.conditional_probability(converted, np.zeros(6, dtype=bool))


def test_bootstrap_is_reproducible_and_ordered() -> None:
    values = np.array([1.0, 2.0, 3.0, 6.0, 12.0])
    first = IMPL.bootstrap_ci(values, np.mean, n_resamples=1_000, random_state=7)
    second = IMPL.bootstrap_ci(values, np.mean, n_resamples=1_000, random_state=7)
    assert first == pytest.approx(second)
    estimate, lower, upper = first
    assert estimate == pytest.approx(values.mean())
    assert lower < estimate < upper


def test_three_way_split_sizes_disjoint_and_exhaustive() -> None:
    train, validation, test = IMPL.three_way_split_indices(100, random_state=11)
    assert (len(train), len(validation), len(test)) == (60, 20, 20)
    combined = np.concatenate([train, validation, test])
    assert len(np.unique(combined)) == 100
    np.testing.assert_array_equal(np.sort(combined), np.arange(100))


def test_three_way_split_reproducible_and_validated() -> None:
    first = IMPL.three_way_split_indices(37, random_state=9)
    second = IMPL.three_way_split_indices(37, random_state=9)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match="sum"):
        IMPL.three_way_split_indices(
            20, train_fraction=0.6, validation_fraction=0.3, test_fraction=0.2
        )


def test_rmse_contract() -> None:
    assert IMPL.root_mean_squared_error(
        np.array([1.0, 2.0]), np.array([2.0, 4.0])
    ) == pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError):
        IMPL.root_mean_squared_error(np.array([1.0]), np.array([1.0, 2.0]))


def test_polynomial_model_fits_quadratic() -> None:
    X = np.linspace(-2.0, 2.0, 50).reshape(-1, 1)
    y = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 0] ** 2
    model = IMPL.build_polynomial_model(2)
    assert isinstance(model, Pipeline)
    model.fit(X, y)
    assert IMPL.root_mean_squared_error(y, model.predict(X)) < 1e-10


def test_degree_evaluation_uses_expected_columns() -> None:
    rng = np.random.default_rng(4)
    X = np.linspace(-2.0, 2.0, 120).reshape(-1, 1)
    y = 1.0 + X[:, 0] - 0.7 * X[:, 0] ** 2 + rng.normal(0, 0.15, len(X))
    results = IMPL.evaluate_degrees(X[:80], y[:80], X[80:], y[80:], [1, 2, 5])
    assert list(results.columns) == ["degree", "train_rmse", "validation_rmse"]
    assert results["degree"].tolist() == [1, 2, 5]
    assert np.isfinite(results[["train_rmse", "validation_rmse"]]).all().all()


def test_best_degree_uses_validation_and_lower_degree_tie_break() -> None:
    results = pd.DataFrame(
        {
            "degree": [7, 2, 4],
            "train_rmse": [0.1, 0.5, 0.2],
            "validation_rmse": [0.8, 0.4, 0.4],
        }
    )
    assert IMPL.select_best_degree(results) == 2


def test_manual_learning_curve_contract() -> None:
    X = np.linspace(-3.0, 3.0, 140).reshape(-1, 1)
    y = 2.0 - X[:, 0] + 0.4 * X[:, 0] ** 2
    curve = IMPL.manual_learning_curve(
        X[:100], y[:100], X[100:], y[100:], degree=2, train_sizes=[20, 50, 100]
    )
    assert list(curve.columns) == ["train_size", "train_rmse", "validation_rmse"]
    assert curve["train_size"].tolist() == [20, 50, 100]
    assert curve["train_rmse"].max() < 1e-9
    assert curve["validation_rmse"].max() < 1e-8


def test_end_to_end_workflow_preserves_holdout_protocol() -> None:
    rng = np.random.default_rng(17)
    X = rng.uniform(-3, 3, size=180).reshape(-1, 1)
    y = 3.0 + 1.5 * X[:, 0] - 0.8 * X[:, 0] ** 2 + rng.normal(0, 0.2, len(X))
    workflow = IMPL.LeakageSafePolynomialWorkflow([1, 2, 7], random_state=8).fit(X, y)
    assert workflow.best_degree_ == IMPL.select_best_degree(workflow.comparison_)
    assert workflow.test_evaluation_count_ == 1
    assert np.isfinite(workflow.test_rmse_)
    indices = np.concatenate(
        [workflow.train_indices_, workflow.validation_indices_, workflow.test_indices_]
    )
    np.testing.assert_array_equal(np.sort(indices), np.arange(len(X)))
    assert workflow.predict(X[:4]).shape == (4,)
    with pytest.raises(RuntimeError, match="fit"):
        IMPL.LeakageSafePolynomialWorkflow([1, 2]).predict(X[:2])
