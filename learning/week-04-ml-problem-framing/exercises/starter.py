"""Starter functions for Exercises 2, 6, 8 and 9.

Complete these without importing from ``solutions``.
"""

from __future__ import annotations

import pandas as pd


def map_scenario_to_task(
    output_kind: str, *, has_labels: bool, deterministic_rules: bool, has_action: bool
) -> str:
    """Return one of classification/regression/clustering/ranking/generation/
    non_ml_rule_engine/not_ready_no_action.
    """
    raise NotImplementedError("TODO Exercise 2")


def audit_prediction_time(
    catalog: pd.DataFrame, *, allowed_offset_hours: float
) -> pd.DataFrame:
    """Return intended features unavailable at prediction time or post-outcome."""
    raise NotImplementedError("TODO Exercise 6")


def expected_error_cost(
    *, false_positives: int, false_negatives: int, fp_cost: float, fn_cost: float
) -> float:
    """Compute FP * fp_cost + FN * fn_cost with input validation."""
    raise NotImplementedError("TODO Exercise 8")


def recommend_delivery_mode(
    *, latency_budget_seconds: float, human_approval_required: bool, data_refresh_hours: float
) -> str:
    """Return batch, online, human_in_loop, or hybrid and document your rule."""
    raise NotImplementedError("TODO Exercise 9")

