"""Reference implementation for Week 3 statistics and model selection."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def _as_1d_float(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a 1-dimensional array")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _as_single_feature_matrix(X: np.ndarray, name: str = "X") -> np.ndarray:
    result = np.asarray(X, dtype=float)
    if result.ndim != 2 or result.shape[1] != 1:
        raise ValueError(f"{name} must have shape (n_samples, 1)")
    if result.shape[0] == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def _validate_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_value = _as_single_feature_matrix(X)
    y_value = _as_1d_float(y, "y")
    if X_value.shape[0] != y_value.size:
        raise ValueError("X rows must match y size")
    return X_value, y_value


def _validate_degree(degree: int) -> int:
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise ValueError("degree must be a positive integer")
    degree_value = int(degree)
    if degree_value < 1 or degree_value > 30:
        raise ValueError("degree must be between 1 and 30")
    return degree_value


def population_summary(values: np.ndarray, ddof: int = 0) -> dict[str, float]:
    data = _as_1d_float(values, "values")
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)):
        raise ValueError("ddof must be a non-negative integer")
    ddof_value = int(ddof)
    if ddof_value < 0 or ddof_value >= data.size:
        raise ValueError("ddof must satisfy 0 <= ddof < number of values")
    return {
        "mean": float(np.mean(data)),
        "variance": float(np.var(data, ddof=ddof_value)),
        "standard_deviation": float(np.std(data, ddof=ddof_value)),
    }


def conditional_probability(event: np.ndarray, given: np.ndarray) -> float:
    event_value = np.asarray(event)
    given_value = np.asarray(given)
    if event_value.ndim != 1 or given_value.ndim != 1:
        raise ValueError("event and given must be 1-dimensional")
    if event_value.shape != given_value.shape or event_value.size == 0:
        raise ValueError("event and given must have the same non-empty shape")
    event_bool = event_value.astype(bool)
    given_bool = given_value.astype(bool)
    denominator = int(np.sum(given_bool))
    if denominator == 0:
        raise ValueError("conditional probability is undefined when given never occurs")
    return float(np.sum(event_bool & given_bool) / denominator)


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    data = _as_1d_float(values, "values")
    if not callable(statistic):
        raise ValueError("statistic must be callable")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    estimate = float(statistic(data))
    if not np.isfinite(estimate):
        raise ValueError("statistic must return a finite scalar")
    rng = np.random.default_rng(random_state)
    distribution = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        resample = data[rng.integers(0, data.size, size=data.size)]
        distribution[index] = float(statistic(resample))
    if not np.isfinite(distribution).all():
        raise ValueError("statistic returned a non-finite bootstrap value")
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(distribution, [alpha, 1.0 - alpha])
    return estimate, float(lower), float(upper)


def three_way_split_indices(
    n_samples: int,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 3:
        raise ValueError("n_samples must be an integer of at least 3")
    fractions = np.array([train_fraction, validation_fraction, test_fraction], dtype=float)
    if not np.isfinite(fractions).all() or np.any(fractions <= 0):
        raise ValueError("all split fractions must be positive and finite")
    if not np.isclose(fractions.sum(), 1.0, atol=1e-12):
        raise ValueError("split fractions must sum to 1")

    n_train = int(np.floor(train_fraction * n_samples))
    n_validation = int(np.floor(validation_fraction * n_samples))
    n_test = n_samples - n_train - n_validation
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError("each split must contain at least one sample")
    permutation = np.random.default_rng(random_state).permutation(n_samples)
    train = permutation[:n_train]
    validation = permutation[n_train : n_train + n_validation]
    test = permutation[n_train + n_validation :]
    return train, validation, test


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = _as_1d_float(y_true, "y_true")
    predicted = _as_1d_float(y_pred, "y_pred")
    if true.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((predicted - true) ** 2)))


def build_polynomial_model(degree: int):
    degree_value = _validate_degree(degree)
    return make_pipeline(
        PolynomialFeatures(degree=degree_value, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )


def evaluate_degrees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    degrees: Iterable[int],
) -> pd.DataFrame:
    X_train_value, y_train_value = _validate_xy(X_train, y_train)
    X_validation_value, y_validation_value = _validate_xy(X_validation, y_validation)
    degree_values = tuple(_validate_degree(degree) for degree in degrees)
    if not degree_values or len(set(degree_values)) != len(degree_values):
        raise ValueError("degrees must be non-empty and unique")

    rows: list[dict[str, float | int]] = []
    for degree in degree_values:
        model = build_polynomial_model(degree)
        model.fit(X_train_value, y_train_value)
        rows.append(
            {
                "degree": degree,
                "train_rmse": root_mean_squared_error(
                    y_train_value, model.predict(X_train_value)
                ),
                "validation_rmse": root_mean_squared_error(
                    y_validation_value, model.predict(X_validation_value)
                ),
            }
        )
    return pd.DataFrame(rows, columns=["degree", "train_rmse", "validation_rmse"])


def select_best_degree(results: pd.DataFrame) -> int:
    required = {"degree", "validation_rmse"}
    if not isinstance(results, pd.DataFrame) or not required.issubset(results.columns):
        raise ValueError("results must contain degree and validation_rmse columns")
    if results.empty:
        raise ValueError("results must not be empty")
    clean = results.loc[:, ["degree", "validation_rmse"]].copy()
    if not np.isfinite(clean["validation_rmse"].to_numpy(dtype=float)).all():
        raise ValueError("validation_rmse must be finite")
    clean["degree"] = clean["degree"].map(_validate_degree)
    ordered = clean.sort_values(["validation_rmse", "degree"], kind="stable")
    return int(ordered.iloc[0]["degree"])


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
    X_train_value, y_train_value = _validate_xy(X_train, y_train)
    X_validation_value, y_validation_value = _validate_xy(X_validation, y_validation)
    degree_value = _validate_degree(degree)
    sizes = tuple(train_sizes)
    if not sizes:
        raise ValueError("train_sizes must not be empty")
    normalized_sizes: list[int] = []
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise ValueError("train sizes must be integers")
        size_value = int(size)
        if size_value < degree_value + 1 or size_value > len(y_train_value):
            raise ValueError("each train size must be > degree and <= training rows")
        normalized_sizes.append(size_value)
    if len(set(normalized_sizes)) != len(normalized_sizes):
        raise ValueError("train_sizes must be unique")

    permutation = np.random.default_rng(random_state).permutation(len(y_train_value))
    rows: list[dict[str, float | int]] = []
    for size in sorted(normalized_sizes):
        subset = permutation[:size]
        model = build_polynomial_model(degree_value)
        model.fit(X_train_value[subset], y_train_value[subset])
        rows.append(
            {
                "train_size": size,
                "train_rmse": root_mean_squared_error(
                    y_train_value[subset], model.predict(X_train_value[subset])
                ),
                "validation_rmse": root_mean_squared_error(
                    y_validation_value, model.predict(X_validation_value)
                ),
            }
        )
    return pd.DataFrame(rows, columns=["train_size", "train_rmse", "validation_rmse"])


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
        X_value, y_value = _validate_xy(X, y)
        train, validation, test = three_way_split_indices(
            len(y_value),
            train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            random_state=self.random_state,
        )
        comparison = evaluate_degrees(
            X_value[train],
            y_value[train],
            X_value[validation],
            y_value[validation],
            self.degrees,
        )
        best_degree = select_best_degree(comparison)
        train_validation = np.concatenate([train, validation])
        final_model = build_polynomial_model(best_degree)
        final_model.fit(X_value[train_validation], y_value[train_validation])
        test_rmse = root_mean_squared_error(y_value[test], final_model.predict(X_value[test]))

        self.train_indices_ = train.copy()
        self.validation_indices_ = validation.copy()
        self.test_indices_ = test.copy()
        self.comparison_ = comparison
        self.best_degree_ = best_degree
        self.model_ = final_model
        self.test_rmse_ = test_rmse
        self.test_evaluation_count_ = 1
        self.n_features_in_ = 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict")
        X_value = _as_single_feature_matrix(X)
        return np.asarray(self.model_.predict(X_value), dtype=float)
