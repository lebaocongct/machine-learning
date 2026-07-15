"""Public tests for the Week 2 gradient-descent challenge."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest


IMPL = importlib.import_module(
    os.getenv("WEEK2_IMPL", "challenge.submission")
)


def test_add_bias_column() -> None:
    X = np.array([[2.0, 3.0], [4.0, 5.0]])
    result = IMPL.add_bias_column(X)
    np.testing.assert_allclose(result, [[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
    assert result.shape == (2, 3)


def test_prediction_and_mse() -> None:
    X_bias = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    theta = np.array([2.0, 3.0])
    predicted = IMPL.predict_linear(X_bias, theta)
    np.testing.assert_allclose(predicted, [5.0, 8.0, 11.0])
    assert IMPL.mean_squared_error(np.array([4.0, 8.0, 12.0]), predicted) == pytest.approx(2 / 3)


def test_analytical_gradient_known_case() -> None:
    X_bias = np.array([[1.0, 1.0], [1.0, 2.0]])
    y = np.array([3.0, 5.0])
    theta = np.zeros(2)
    np.testing.assert_allclose(IMPL.mse_gradient(X_bias, y, theta), [-8.0, -13.0])


def test_numerical_gradient_matches_analytical() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 3))
    X_bias = IMPL.add_bias_column(X)
    y = rng.normal(size=12)
    theta = rng.normal(size=4)
    analytical = IMPL.mse_gradient(X_bias, y, theta)
    numerical = IMPL.finite_difference_gradient(
        lambda candidate: IMPL.mean_squared_error(
            y, IMPL.predict_linear(X_bias, candidate)
        ),
        theta,
    )
    relative_error = np.linalg.norm(analytical - numerical) / max(
        1.0, np.linalg.norm(analytical) + np.linalg.norm(numerical)
    )
    assert relative_error < 1e-6


def test_standardizer() -> None:
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    mean, scale = IMPL.fit_standardizer(X)
    standardized = IMPL.apply_standardizer(X, mean, scale)
    np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(standardized.std(axis=0), 1.0, atol=1e-12)


def test_constant_feature_is_safe() -> None:
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    mean, scale = IMPL.fit_standardizer(X)
    standardized = IMPL.apply_standardizer(X, mean, scale)
    assert scale[1] == pytest.approx(1.0)
    np.testing.assert_allclose(standardized[:, 1], 0.0)


def test_gradient_descent_loss_decreases() -> None:
    x = np.linspace(-2.0, 2.0, 40).reshape(-1, 1)
    y = 2.0 + 3.0 * x[:, 0]
    X_bias = IMPL.add_bias_column(x)
    theta, history = IMPL.gradient_descent(
        X_bias, y, np.zeros(2), learning_rate=0.05, n_steps=500
    )
    assert history.shape == (501,)
    assert history[-1] < 1e-10
    assert history[-1] < history[0]
    np.testing.assert_allclose(theta, [2.0, 3.0], atol=1e-5)


def test_model_recovers_noiseless_parameters() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(120, 3)) * np.array([1.0, 10.0, 100.0])
    true_intercept = -4.0
    true_coef = np.array([2.0, -0.5, 0.03])
    y = true_intercept + X @ true_coef
    model = IMPL.LinearRegressionGD(learning_rate=0.05, n_steps=1_500).fit(X, y)
    np.testing.assert_allclose(model.coef_, true_coef, atol=1e-5)
    assert model.intercept_ == pytest.approx(true_intercept, abs=1e-5)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-5)


def test_model_does_not_mutate_inputs() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([3.0, 5.0, 7.0])
    X_before = X.copy()
    y_before = y.copy()
    IMPL.LinearRegressionGD(n_steps=100).fit(X, y)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


def test_predict_before_fit_raises() -> None:
    model = IMPL.LinearRegressionGD()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.ones((2, 1)))


def test_invalid_shapes_raise() -> None:
    with pytest.raises(ValueError):
        IMPL.add_bias_column(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        IMPL.mean_squared_error(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        IMPL.mse_gradient(np.ones((3, 2)), np.ones(4), np.ones(2))


def test_invalid_hyperparameters_raise() -> None:
    with pytest.raises(ValueError):
        IMPL.LinearRegressionGD(learning_rate=0.0)
    with pytest.raises(ValueError):
        IMPL.LinearRegressionGD(learning_rate=np.nan)
    with pytest.raises(ValueError):
        IMPL.LinearRegressionGD(n_steps=0)
    with pytest.raises(ValueError):
        IMPL.LinearRegressionGD(n_steps=True)
    with pytest.raises(ValueError):
        IMPL.finite_difference_gradient(lambda theta: 0.0, np.ones(2), epsilon=np.nan)

    huge_design = IMPL.add_bias_column(np.array([[1e150]]))
    with pytest.raises(FloatingPointError, match="non-finite"):
        IMPL.gradient_descent(
            huge_design,
            np.array([1.0]),
            np.zeros(2),
            learning_rate=1.0,
            n_steps=5,
        )
