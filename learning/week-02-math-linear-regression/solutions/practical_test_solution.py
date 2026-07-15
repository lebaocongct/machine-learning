"""Reference implementation for the Week 2 practical test."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solutions.exercise_solutions import r2_score_numpy, root_mean_squared_error
from solutions.linear_regression_solution import (
    LinearRegressionGD,
    add_bias_column,
    apply_standardizer,
    finite_difference_gradient,
    fit_standardizer,
    gradient_descent,
    mean_squared_error,
    mse_gradient,
    predict_linear,
)


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["study_hours", "practice_tests", "sleep_hours", "absences"]


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def gradient_check(X_bias: np.ndarray, y: np.ndarray) -> float:
    theta = np.array([5.0, 1.0, -1.0, 0.5, 2.0])
    analytical = mse_gradient(X_bias, y, theta)
    numerical = finite_difference_gradient(
        lambda candidate: mean_squared_error(
            y, predict_linear(X_bias, candidate)
        ),
        theta,
    )
    return float(
        np.linalg.norm(analytical - numerical)
        / max(1.0, np.linalg.norm(analytical) + np.linalg.norm(numerical))
    )


def run_learning_rates(
    X_bias: np.ndarray, y: np.ndarray
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    rows = []
    histories: dict[float, np.ndarray] = {}
    for learning_rate in [0.001, 0.01, 0.05, 0.2, 1.0]:
        try:
            _, history = gradient_descent(
                X_bias,
                y,
                np.zeros(X_bias.shape[1]),
                learning_rate,
                500,
            )
            histories[learning_rate] = history
            rows.append(
                {
                    "learning_rate": learning_rate,
                    "initial_loss": history[0],
                    "final_loss": history[-1],
                    "monotonic": bool(np.all(np.diff(history) <= 1e-9)),
                    "finite": bool(np.isfinite(history).all()),
                }
            )
        except FloatingPointError:
            rows.append(
                {
                    "learning_rate": learning_rate,
                    "initial_loss": np.nan,
                    "final_loss": np.inf,
                    "monotonic": False,
                    "finite": False,
                }
            )
    return pd.DataFrame(rows), histories


def save_figures(
    histories: dict[float, np.ndarray],
    y: np.ndarray,
    predictions: np.ndarray,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for learning_rate, history in histories.items():
        ax.plot(history, label=f"lr={learning_rate}")
    ax.set_yscale("log")
    ax.set(title="Practical: loss by learning rate", xlabel="Step", ylabel="MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loss_by_learning_rate.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, predictions, alpha=0.7)
    bounds = [min(y.min(), predictions.min()), max(y.max(), predictions.max())]
    ax.plot(bounds, bounds, linestyle="--", color="black")
    ax.set(title="Predicted vs actual score", xlabel="Actual", ylabel="Predicted")
    fig.tight_layout()
    fig.savefig(output / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def main() -> None:
    output = ROOT / "outputs" / "reference_practical"
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(ROOT / "data" / "practical_regression.csv")
    X = frame[FEATURES].to_numpy(dtype=float)
    y = frame["score"].to_numpy(dtype=float)
    mean, scale = fit_standardizer(X)
    X_bias = add_bias_column(apply_standardizer(X, mean, scale))

    relative_error = gradient_check(X_bias, y)
    experiments, histories = run_learning_rates(X_bias, y)
    model = LinearRegressionGD(learning_rate=0.05, n_steps=2_000).fit(X, y)
    predictions = model.predict(X)

    metrics = {
        "gradient_relative_error": relative_error,
        "mse": mean_squared_error(y, predictions),
        "rmse": root_mean_squared_error(y, predictions),
        "mae": mean_absolute_error(y, predictions),
        "r2": r2_score_numpy(y, predictions),
        "intercept": model.intercept_,
        "coefficients": dict(zip(FEATURES, model.coef_, strict=True)),
    }

    assert relative_error < 1e-6
    assert metrics["mse"] < np.var(y)
    assert np.isfinite(model.loss_history_).all()

    pd.DataFrame(
        {
            "student_id": frame["student_id"],
            "actual": y,
            "predicted": predictions,
            "residual": predictions - y,
        }
    ).to_csv(output / "practical_predictions.csv", index=False)
    experiments.to_csv(output / "learning_rate_experiment.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    save_figures(histories, y, predictions, output)

    print(json.dumps(metrics, indent=2))
    print(experiments.to_string(index=False))


if __name__ == "__main__":
    main()

