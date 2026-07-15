"""Reproducible reference pipeline for the Week 4 fraud practical."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .problem_framing_solution import (
        CostSensitiveHoldoutWorkflow,
        audit_feature_catalog,
        binary_classification_metrics,
    )
except ImportError:  # Allow `python solutions/practical_test_solution.py`.
    from problem_framing_solution import (
        CostSensitiveHoldoutWorkflow,
        audit_feature_catalog,
        binary_classification_metrics,
    )


FP_COST = 8.0
FN_COST = 300.0
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def slice_report(
    test: pd.DataFrame, *, target: str, prediction: str, columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        for value, group in test.groupby(column, dropna=False):
            metrics = binary_classification_metrics(
                group[target].to_numpy(), group[prediction].to_numpy()
            )
            rows.append(
                {
                    "slice_column": column,
                    "slice_value": value,
                    "n": len(group),
                    "prevalence": float(group[target].mean()),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "predicted_positive_rate": float(group[prediction].mean()),
                    "total_cost": metrics["fp"] * FP_COST + metrics["fn"] * FN_COST,
                }
            )
    return pd.DataFrame(rows)


def run(data_dir: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_dir / "fraud_review.csv")
    catalog = pd.read_csv(data_dir / "fraud_feature_catalog.csv")

    risks = audit_feature_catalog(catalog, max_allowed_offset_hours=0.02)
    risks.to_csv(output_dir / "fraud-feature-audit.csv", index=False)

    workflow = CostSensitiveHoldoutWorkflow(
        THRESHOLDS,
        false_positive_cost=FP_COST,
        false_negative_cost=FN_COST,
    ).fit(
        data["heuristic_risk_score"].to_numpy(),
        data["confirmed_fraud"].to_numpy(),
        data["event_time"],
    )
    workflow.validation_results_.to_csv(
        output_dir / "fraud-threshold-sweep.csv", index=False
    )

    target = data["confirmed_fraud"].to_numpy()
    prevalence = {
        "train": float(target[workflow.train_indices_].mean()),
        "validation": float(target[workflow.validation_indices_].mean()),
        "test": float(target[workflow.test_indices_].mean()),
    }
    result: dict[str, object] = {
        "split_sizes": {
            "train": len(workflow.train_indices_),
            "validation": len(workflow.validation_indices_),
            "test": len(workflow.test_indices_),
        },
        "prevalence": prevalence,
        "selected_threshold": workflow.selected_threshold_,
        **workflow.test_metrics_,
        "test_evaluation_count": workflow.test_evaluation_count_,
        "majority_test_baseline": workflow.majority_test_baseline_,
    }
    with (output_dir / "fraud-test-metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    test = data.iloc[workflow.test_indices_].copy()
    test["prediction"] = workflow.predict(test["heuristic_risk_score"].to_numpy())
    slices = slice_report(
        test,
        target="confirmed_fraud",
        prediction="prediction",
        columns=["country_risk", "card_present"],
    )
    slices.to_csv(output_dir / "fraud-slices.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/reference-practical")
    )
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
