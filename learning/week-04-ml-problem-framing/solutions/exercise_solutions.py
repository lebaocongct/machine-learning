"""Reference functions for selected Week 4 exercises."""

from __future__ import annotations

import math

import pandas as pd


def map_scenario_to_task(
    output_kind: str, *, has_labels: bool, deterministic_rules: bool, has_action: bool
) -> str:
    """Map a scenario to the simplest defensible task family."""
    if not has_action:
        return "not_ready_no_action"
    if deterministic_rules:
        return "non_ml_rule_engine"
    kind = output_kind.strip().lower()
    if kind in {"binary", "categorical", "category", "multiclass"}:
        if not has_labels:
            raise ValueError("classification requires labels")
        return "classification"
    if kind in {"numeric", "continuous", "number"}:
        if not has_labels:
            raise ValueError("regression requires labels")
        return "regression"
    if kind in {"clusters", "groups", "segments"}:
        if has_labels:
            raise ValueError("clustering assumes no predefined labels")
        return "clustering"
    if kind in {"ordered_list", "ranking", "relevance_order"}:
        return "ranking"
    if kind in {"generated_content", "text_generation", "image_generation"}:
        return "generation"
    raise ValueError(f"unsupported output kind: {output_kind}")


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError("intended_for_model must be boolean")


def audit_prediction_time(
    catalog: pd.DataFrame, *, allowed_offset_hours: float
) -> pd.DataFrame:
    required = {
        "feature_name",
        "source_stage",
        "available_offset_hours",
        "intended_for_model",
    }
    if not required.issubset(catalog.columns):
        raise ValueError(f"missing columns: {sorted(required - set(catalog.columns))}")
    if not math.isfinite(allowed_offset_hours) or allowed_offset_hours < 0:
        raise ValueError("allowed_offset_hours must be non-negative")

    rows: list[dict[str, str]] = []
    for _, row in catalog.iterrows():
        if not _to_bool(row["intended_for_model"]):
            continue
        stage = str(row["source_stage"]).strip().lower()
        offset = float(row["available_offset_hours"])
        if stage in {"post_outcome", "label_generation"}:
            rows.append(
                {
                    "feature_name": str(row["feature_name"]),
                    "risk_type": "post_outcome_leakage",
                    "reason": f"source_stage={stage}",
                }
            )
        elif not math.isfinite(offset) or offset > allowed_offset_hours:
            rows.append(
                {
                    "feature_name": str(row["feature_name"]),
                    "risk_type": "unavailable_at_prediction",
                    "reason": f"available +{offset:g}h; allowed +{allowed_offset_hours:g}h",
                }
            )
    return pd.DataFrame(rows, columns=["feature_name", "risk_type", "reason"])


def expected_error_cost(
    *, false_positives: int, false_negatives: int, fp_cost: float, fn_cost: float
) -> float:
    values = (false_positives, false_negatives, fp_cost, fn_cost)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("counts and costs must be finite numbers")
    if false_positives < 0 or false_negatives < 0 or fp_cost < 0 or fn_cost < 0:
        raise ValueError("counts and costs must be non-negative")
    if int(false_positives) != false_positives or int(false_negatives) != false_negatives:
        raise ValueError("error counts must be integers")
    return float(false_positives * fp_cost + false_negatives * fn_cost)


def recommend_delivery_mode(
    *, latency_budget_seconds: float, human_approval_required: bool, data_refresh_hours: float
) -> str:
    """A documented teaching heuristic, not a universal architecture rule."""
    if latency_budget_seconds <= 0 or data_refresh_hours <= 0:
        raise ValueError("latency and refresh must be positive")
    if human_approval_required:
        return "hybrid" if latency_budget_seconds <= 1 else "human_in_loop"
    if latency_budget_seconds <= 1:
        return "online"
    if data_refresh_hours >= 6:
        return "batch"
    return "hybrid"


if __name__ == "__main__":
    assert map_scenario_to_task(
        "binary", has_labels=True, deterministic_rules=False, has_action=True
    ) == "classification"
    assert map_scenario_to_task(
        "numeric", has_labels=True, deterministic_rules=True, has_action=True
    ) == "non_ml_rule_engine"
    assert expected_error_cost(
        false_positives=220, false_negatives=20, fp_cost=10, fn_cost=200
    ) == 6200
    assert recommend_delivery_mode(
        latency_budget_seconds=0.15,
        human_approval_required=False,
        data_refresh_hours=0.01,
    ) == "online"
    print("Reference exercise assertions passed.")

