"""Reference implementation for Week 4 problem framing."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


CANVAS_STRING_FIELDS = (
    "problem_name",
    "non_ml_goal",
    "decision",
    "decision_owner",
    "prediction_unit",
    "prediction_time",
    "target",
    "target_window",
    "model_output",
    "primary_metric",
    "baseline",
    "success_criterion",
)
CANVAS_LIST_FIELDS = ("constraints", "evaluation_slices", "leakage_risks", "non_goals")
TASK_TYPES = {"classification", "regression", "clustering", "ranking", "generation"}
DEPLOYMENT_MODES = {"batch", "online", "human_in_loop", "hybrid"}


def _as_1d(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return result


def _as_binary(values: np.ndarray, name: str) -> np.ndarray:
    result = _as_1d(values, name)
    try:
        numeric = result.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only binary 0/1 values") from exc
    if not np.isin(numeric, [0, 1]).all() or not np.array_equal(result, numeric):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return numeric


def _as_float_1d(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite non-empty 1D array")
    return result


def _to_bool(value: object, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{field} must contain boolean values")


def classify_task(output_kind: str, has_labels: bool) -> str:
    if not isinstance(output_kind, str) or not output_kind.strip():
        raise ValueError("output_kind must be a non-empty string")
    kind = output_kind.strip().lower()
    if not isinstance(has_labels, (bool, np.bool_)):
        raise ValueError("has_labels must be boolean")
    if kind in {"binary", "categorical", "category", "multiclass"}:
        if not has_labels:
            raise ValueError("classification requires labeled examples")
        return "classification"
    if kind in {"numeric", "number", "continuous"}:
        if not has_labels:
            raise ValueError("regression requires labeled examples")
        return "regression"
    if kind in {"clusters", "groups", "segments"}:
        if has_labels:
            raise ValueError("clustering framing expects no predefined labels")
        return "clustering"
    if kind in {"ordered_list", "ranking", "relevance_order"}:
        return "ranking"
    if kind in {"generated_content", "text_generation", "image_generation"}:
        return "generation"
    raise ValueError(f"unsupported output_kind: {output_kind}")


def validate_problem_canvas(canvas: dict[str, object]) -> list[str]:
    if not isinstance(canvas, dict):
        return ["canvas must be a dictionary"]
    errors: list[str] = []
    for field in CANVAS_STRING_FIELDS:
        value = canvas.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")
    for field in CANVAS_LIST_FIELDS:
        value = canvas.get(field)
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            errors.append(f"{field} must be a non-empty list of strings")
    if canvas.get("task_type") not in TASK_TYPES:
        errors.append(f"task_type must be one of {sorted(TASK_TYPES)}")
    if canvas.get("deployment_mode") not in DEPLOYMENT_MODES:
        errors.append(f"deployment_mode must be one of {sorted(DEPLOYMENT_MODES)}")
    return sorted(errors)


def validate_prediction_contract(contract: dict[str, object]) -> list[str]:
    if not isinstance(contract, dict):
        return ["contract must be a dictionary"]
    errors: list[str] = []
    string_fields = (
        "unit_of_prediction",
        "prediction_time",
        "target_name",
        "target_definition",
        "output_type",
        "owner",
    )
    for field in string_fields:
        value = contract.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")
    if contract.get("output_type") not in {"binary", "numeric", "categorical", "ranking_score"}:
        errors.append("output_type is unsupported")
    try:
        start = float(contract.get("target_window_start_hours"))
        end = float(contract.get("target_window_end_hours"))
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            errors.append("target window must satisfy 0 <= start < end")
    except (TypeError, ValueError):
        errors.append("target window hours must be numeric")
    try:
        latency = float(contract.get("latency_sla_ms"))
        if not np.isfinite(latency) or latency <= 0:
            errors.append("latency_sla_ms must be positive")
    except (TypeError, ValueError):
        errors.append("latency_sla_ms must be numeric")
    return sorted(set(errors))


def audit_feature_catalog(
    catalog: pd.DataFrame, *, max_allowed_offset_hours: float = 0.0
) -> pd.DataFrame:
    required = {
        "feature_name",
        "source_stage",
        "available_offset_hours",
        "intended_for_model",
    }
    if not isinstance(catalog, pd.DataFrame) or not required.issubset(catalog.columns):
        raise ValueError(f"catalog must contain columns {sorted(required)}")
    if not np.isfinite(max_allowed_offset_hours) or max_allowed_offset_hours < 0:
        raise ValueError("max_allowed_offset_hours must be non-negative and finite")

    risks: list[dict[str, str]] = []
    duplicate_names = set(
        catalog.loc[catalog["feature_name"].duplicated(keep=False), "feature_name"].astype(str)
    )
    for _, row in catalog.iterrows():
        name = str(row["feature_name"]).strip()
        stage = str(row["source_stage"]).strip().lower()
        try:
            offset = float(row["available_offset_hours"])
        except (TypeError, ValueError) as exc:
            raise ValueError("available_offset_hours must be numeric") from exc
        intended = _to_bool(row["intended_for_model"], "intended_for_model")
        if not name:
            risks.append({"feature_name": name, "risk_type": "invalid_name", "reason": "empty name"})
        if name in duplicate_names:
            risks.append(
                {"feature_name": name, "risk_type": "duplicate_feature", "reason": "duplicate catalog entry"}
            )
        if intended and stage in {"post_outcome", "label_generation"}:
            risks.append(
                {
                    "feature_name": name,
                    "risk_type": "post_outcome_leakage",
                    "reason": f"source_stage={stage}",
                }
            )
        elif intended and (not np.isfinite(offset) or offset > max_allowed_offset_hours):
            risks.append(
                {
                    "feature_name": name,
                    "risk_type": "unavailable_at_prediction",
                    "reason": f"available at +{offset:g}h; allowed +{max_allowed_offset_hours:g}h",
                }
            )
    return pd.DataFrame(risks, columns=["feature_name", "risk_type", "reason"])


def time_ordered_split_indices(
    timestamps: Iterable[object],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_values = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce")
    n = len(time_values)
    if n < 3 or time_values.isna().any():
        raise ValueError("timestamps must contain at least three valid values")
    fractions = np.array([train_fraction, validation_fraction], dtype=float)
    if not np.isfinite(fractions).all() or np.any(fractions <= 0) or fractions.sum() >= 1:
        raise ValueError("train/validation fractions must be positive with sum < 1")
    order = np.argsort(time_values.to_numpy(), kind="stable")
    n_train = int(np.floor(train_fraction * n))
    n_validation = int(np.floor(validation_fraction * n))
    n_test = n - n_train - n_validation
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError("each split must contain at least one row")
    return (
        order[:n_train].astype(int),
        order[n_train : n_train + n_validation].astype(int),
        order[n_train + n_validation :].astype(int),
    )


def regression_baseline(
    y_train: np.ndarray, y_evaluation: np.ndarray, *, strategy: str = "mean"
) -> dict[str, float]:
    train = _as_float_1d(y_train, "y_train")
    evaluation = _as_float_1d(y_evaluation, "y_evaluation")
    if strategy == "mean":
        constant = float(np.mean(train))
    elif strategy == "median":
        constant = float(np.median(train))
    else:
        raise ValueError("strategy must be 'mean' or 'median'")
    residual = evaluation - constant
    return {
        "constant": constant,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def binary_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float | int]:
    true = _as_binary(y_true, "y_true")
    predicted = _as_binary(y_pred, "y_pred")
    if true.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    tp = int(np.sum((true == 1) & (predicted == 1)))
    fp = int(np.sum((true == 0) & (predicted == 1)))
    tn = int(np.sum((true == 0) & (predicted == 0)))
    fn = int(np.sum((true == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float((tp + tn) / len(true)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def majority_classification_baseline(
    y_train: np.ndarray, y_evaluation: np.ndarray
) -> dict[str, float | int]:
    train = _as_binary(y_train, "y_train")
    evaluation = _as_binary(y_evaluation, "y_evaluation")
    counts = np.bincount(train, minlength=2)
    majority_class = int(np.argmax(counts))
    predictions = np.full(len(evaluation), majority_class, dtype=int)
    result = binary_classification_metrics(evaluation, predictions)
    return {"majority_class": majority_class, **result}


def threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Iterable[float],
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> pd.DataFrame:
    true = _as_binary(y_true, "y_true")
    score_values = _as_float_1d(scores, "scores")
    if true.shape != score_values.shape:
        raise ValueError("y_true and scores must have the same shape")
    if np.any((score_values < 0) | (score_values > 1)):
        raise ValueError("scores must be between 0 and 1")
    if (
        not np.isfinite([false_positive_cost, false_negative_cost]).all()
        or false_positive_cost < 0
        or false_negative_cost < 0
    ):
        raise ValueError("error costs must be non-negative and finite")
    threshold_values = tuple(float(value) for value in thresholds)
    if (
        not threshold_values
        or len(set(threshold_values)) != len(threshold_values)
        or not np.isfinite(threshold_values).all()
        or any(value < 0 or value > 1 for value in threshold_values)
    ):
        raise ValueError("thresholds must be unique finite values in [0, 1]")

    rows: list[dict[str, float | int]] = []
    for threshold in threshold_values:
        metrics = binary_classification_metrics(true, (score_values >= threshold).astype(int))
        rows.append(
            {
                "threshold": threshold,
                **metrics,
                "total_cost": float(
                    metrics["fp"] * false_positive_cost
                    + metrics["fn"] * false_negative_cost
                ),
            }
        )
    columns = [
        "threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "total_cost",
    ]
    return pd.DataFrame(rows, columns=columns)


def select_operating_threshold(results: pd.DataFrame) -> float:
    required = {"threshold", "total_cost", "recall"}
    if not isinstance(results, pd.DataFrame) or not required.issubset(results.columns):
        raise ValueError(f"results must contain {sorted(required)}")
    if results.empty or not np.isfinite(
        results[["threshold", "total_cost", "recall"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("results must be non-empty and finite")
    ordered = results.sort_values(
        ["total_cost", "recall", "threshold"],
        ascending=[True, False, True],
        kind="stable",
    )
    return float(ordered.iloc[0]["threshold"])


class CostSensitiveHoldoutWorkflow:
    """Select a score threshold on validation and evaluate test once."""

    def __init__(
        self,
        thresholds: Iterable[float],
        *,
        false_positive_cost: float,
        false_negative_cost: float,
        train_fraction: float = 0.70,
        validation_fraction: float = 0.15,
    ) -> None:
        self.thresholds = tuple(thresholds)
        self.false_positive_cost = false_positive_cost
        self.false_negative_cost = false_negative_cost
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction

    def fit(
        self, scores: np.ndarray, y: np.ndarray, timestamps: Iterable[object]
    ) -> "CostSensitiveHoldoutWorkflow":
        score_values = _as_float_1d(scores, "scores")
        y_value = _as_binary(y, "y")
        timestamp_values = list(timestamps)
        if len(score_values) != len(y_value) or len(timestamp_values) != len(y_value):
            raise ValueError("scores, y and timestamps must have the same length")
        train, validation, test = time_ordered_split_indices(
            timestamp_values,
            train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction,
        )
        validation_results = threshold_sweep(
            y_value[validation],
            score_values[validation],
            self.thresholds,
            false_positive_cost=self.false_positive_cost,
            false_negative_cost=self.false_negative_cost,
        )
        selected = select_operating_threshold(validation_results)
        test_predictions = (score_values[test] >= selected).astype(int)
        test_metrics = binary_classification_metrics(y_value[test], test_predictions)
        test_metrics["total_cost"] = float(
            test_metrics["fp"] * self.false_positive_cost
            + test_metrics["fn"] * self.false_negative_cost
        )

        self.train_indices_ = train.copy()
        self.validation_indices_ = validation.copy()
        self.test_indices_ = test.copy()
        self.validation_results_ = validation_results
        self.selected_threshold_ = selected
        self.test_metrics_ = test_metrics
        self.majority_test_baseline_ = majority_classification_baseline(
            y_value[train], y_value[test]
        )
        self.test_evaluation_count_ = 1
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not hasattr(self, "selected_threshold_"):
            raise RuntimeError("fit must be called before predict")
        score_values = _as_float_1d(scores, "scores")
        if np.any((score_values < 0) | (score_values > 1)):
            raise ValueError("scores must be between 0 and 1")
        return (score_values >= self.selected_threshold_).astype(int)
