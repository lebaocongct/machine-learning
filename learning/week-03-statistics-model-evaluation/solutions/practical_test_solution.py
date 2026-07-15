"""Reference practical solution for Week 3.

Run only after submitting the timed practical:

    python -m solutions.practical_test_solution
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solutions.model_evaluation_solution import (
    bootstrap_ci,
    build_polynomial_model,
    evaluate_degrees,
    manual_learning_curve,
    root_mean_squared_error,
    select_best_degree,
    three_way_split_indices,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "reference_practical"


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual_sum = np.sum((y_true - y_pred) ** 2)
    total_sum = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - residual_sum / total_sum)


def assert_split(train: np.ndarray, validation: np.ndarray, test: np.ndarray, n: int) -> None:
    combined = np.concatenate([train, validation, test])
    assert len(np.unique(combined)) == n
    np.testing.assert_array_equal(np.sort(combined), np.arange(n))


def save_degree_plot(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["degree"], results["train_rmse"], "o-", label="Train")
    ax.plot(results["degree"], results["validation_rmse"], "o-", label="Validation")
    ax.set(xlabel="Polynomial degree", ylabel="RMSE", title="Validation curve")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "degree_validation_curve.png", dpi=150)
    plt.close(fig)


def save_learning_curve(curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["train_size"], curve["train_rmse"], "o-", label="Train")
    ax.plot(curve["train_size"], curve["validation_rmse"], "o-", label="Validation")
    ax.set(xlabel="Training rows", ylabel="RMSE", title="Learning curve")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "learning_curve.png", dpi=150)
    plt.close(fig)


def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7)
    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(limits, limits, "--", color="black", label="Perfect")
    ax.set(xlabel="Actual y", ylabel="Predicted y", title="Practical: predicted vs actual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "data" / "practical_model_selection.csv")
    feature_columns = ["x"]
    assert feature_columns == ["x"]
    assert "post_event_target_proxy" not in feature_columns
    X = frame[feature_columns].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)

    train, validation, test = three_way_split_indices(len(frame), random_state=73)
    assert_split(train, validation, test, len(frame))

    comparison = evaluate_degrees(
        X[train], y[train], X[validation], y[validation], range(1, 13)
    )
    selected_degree = select_best_degree(comparison)
    comparison.to_csv(OUTPUT / "practical_degree_comparison.csv", index=False)
    save_degree_plot(comparison)

    learning_curve = manual_learning_curve(
        X[train],
        y[train],
        X[validation],
        y[validation],
        degree=selected_degree,
        train_sizes=[30, 60, 90, 120, len(train)],
        random_state=73,
    )
    learning_curve.to_csv(OUTPUT / "practical_learning_curve.csv", index=False)
    save_learning_curve(learning_curve)

    # Model decision is now frozen. Only after this point is test y accessed.
    development = np.concatenate([train, validation])
    final_model = build_polynomial_model(selected_degree)
    final_model.fit(X[development], y[development])
    test_predictions = final_model.predict(X[test])
    test_y = y[test]

    absolute_errors = np.abs(test_predictions - test_y)
    metrics = {
        "selected_degree": selected_degree,
        "split_sizes": [len(train), len(validation), len(test)],
        "test_rmse": root_mean_squared_error(test_y, test_predictions),
        "test_mae": float(np.mean(absolute_errors)),
        "test_r2": r2_score_manual(test_y, test_predictions),
    }
    estimate, lower, upper = bootstrap_ci(
        absolute_errors, np.mean, n_resamples=4_000, random_state=99
    )
    metrics["mae_bootstrap_estimate"] = estimate
    metrics["mae_ci_95"] = [lower, upper]

    quality = frame["sensor_quality"].to_numpy(dtype=float)[test]
    quality_cutoff = float(np.median(quality))
    low = quality < quality_cutoff
    high = ~low
    metrics["sensor_quality_median"] = quality_cutoff
    metrics["low_quality_rmse"] = root_mean_squared_error(
        test_y[low], test_predictions[low]
    )
    metrics["high_quality_rmse"] = root_mean_squared_error(
        test_y[high], test_predictions[high]
    )

    predictions = pd.DataFrame(
        {
            "observation_id": frame.iloc[test]["observation_id"].to_numpy(),
            "actual_y": test_y,
            "predicted_y": test_predictions,
            "absolute_error": absolute_errors,
            "sensor_quality": quality,
        }
    )
    predictions.to_csv(OUTPUT / "practical_predictions.csv", index=False)
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_prediction_plot(test_y, test_predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
