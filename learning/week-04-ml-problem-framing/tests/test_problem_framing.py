"""Public tests for the Week 4 problem-framing challenge."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd
import pytest


IMPL = importlib.import_module(os.getenv("WEEK4_IMPL", "challenge.submission"))


def valid_canvas() -> dict[str, object]:
    return {
        "problem_name": "Support escalation triage",
        "non_ml_goal": "Route risky tickets early",
        "decision": "Specialist or standard queue",
        "decision_owner": "Support operations",
        "prediction_unit": "One new ticket",
        "prediction_time": "Ticket creation after parsing",
        "target": "Escalated within 48 hours",
        "target_window": "0 to 48 hours after creation",
        "task_type": "classification",
        "model_output": "Escalation probability",
        "primary_metric": "Total triage error cost",
        "baseline": "Existing heuristic risk score",
        "success_criterion": "At least 15% lower validation cost",
        "deployment_mode": "human_in_loop",
        "constraints": ["p95 latency below 500 ms"],
        "evaluation_slices": ["channel", "customer tier"],
        "leakage_risks": ["post-resolution fields"],
        "non_goals": ["Fully automate support decisions"],
    }


def test_classify_task_families() -> None:
    assert IMPL.classify_task("binary", True) == "classification"
    assert IMPL.classify_task("numeric", True) == "regression"
    assert IMPL.classify_task("clusters", False) == "clustering"
    assert IMPL.classify_task("ordered_list", True) == "ranking"
    assert IMPL.classify_task("generated_content", False) == "generation"


def test_valid_problem_canvas_has_no_errors() -> None:
    assert IMPL.validate_problem_canvas(valid_canvas()) == []


def test_problem_canvas_reports_missing_and_invalid_fields() -> None:
    canvas = valid_canvas()
    canvas["decision"] = ""
    canvas["task_type"] = "magic"
    canvas["constraints"] = []
    errors = IMPL.validate_problem_canvas(canvas)
    assert any("decision" in error for error in errors)
    assert any("task_type" in error for error in errors)
    assert any("constraints" in error for error in errors)


def test_prediction_contract_validation() -> None:
    contract = {
        "unit_of_prediction": "one ticket",
        "prediction_time": "ticket creation",
        "target_name": "escalated_within_48h",
        "target_definition": "binary escalation outcome",
        "target_window_start_hours": 0,
        "target_window_end_hours": 48,
        "output_type": "binary",
        "latency_sla_ms": 500,
        "owner": "support operations",
    }
    assert IMPL.validate_prediction_contract(contract) == []
    contract["target_window_end_hours"] = 0
    assert any("window" in error for error in IMPL.validate_prediction_contract(contract))


def test_feature_catalog_audit_finds_post_outcome_and_late_features() -> None:
    catalog = pd.DataFrame(
        {
            "feature_name": ["past_count", "future_value", "resolution"],
            "source_stage": ["prediction", "prediction", "post_outcome"],
            "available_offset_hours": [0.0, 2.0, 24.0],
            "intended_for_model": [True, True, True],
        }
    )
    risks = IMPL.audit_feature_catalog(catalog)
    risk_by_feature = dict(zip(risks["feature_name"], risks["risk_type"]))
    assert "past_count" not in risk_by_feature
    assert risk_by_feature["future_value"] == "unavailable_at_prediction"
    assert risk_by_feature["resolution"] == "post_outcome_leakage"


def test_time_ordered_split_is_exhaustive_and_chronological() -> None:
    timestamps = pd.date_range("2026-01-01", periods=100, freq="h")[::-1]
    train, validation, test = IMPL.time_ordered_split_indices(timestamps)
    assert (len(train), len(validation), len(test)) == (70, 15, 15)
    combined = np.concatenate([train, validation, test])
    np.testing.assert_array_equal(np.sort(combined), np.arange(100))
    parsed = pd.to_datetime(pd.Series(timestamps))
    assert parsed.iloc[train].max() <= parsed.iloc[validation].min()
    assert parsed.iloc[validation].max() <= parsed.iloc[test].min()


def test_regression_baselines() -> None:
    train = np.array([1.0, 2.0, 9.0])
    evaluation = np.array([2.0, 4.0])
    mean_result = IMPL.regression_baseline(train, evaluation, strategy="mean")
    median_result = IMPL.regression_baseline(train, evaluation, strategy="median")
    assert mean_result["constant"] == pytest.approx(4.0)
    assert mean_result["mae"] == pytest.approx(1.0)
    assert median_result["constant"] == pytest.approx(2.0)


def test_majority_classification_baseline() -> None:
    result = IMPL.majority_classification_baseline(
        np.array([0, 0, 0, 1]), np.array([0, 1, 1])
    )
    assert result["majority_class"] == 0
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.0


def test_binary_metrics_known_confusion_matrix() -> None:
    result = IMPL.binary_classification_metrics(
        np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
    )
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 1, 1)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_threshold_sweep_computes_cost() -> None:
    results = IMPL.threshold_sweep(
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.4, 0.6, 0.1]),
        [0.3, 0.7],
        false_positive_cost=2,
        false_negative_cost=10,
    )
    low = results.iloc[0]
    high = results.iloc[1]
    assert low["total_cost"] == pytest.approx(2.0)
    assert high["total_cost"] == pytest.approx(10.0)


def test_threshold_selection_tie_breaks_on_recall_then_threshold() -> None:
    results = pd.DataFrame(
        {
            "threshold": [0.7, 0.3, 0.5],
            "total_cost": [20.0, 20.0, 30.0],
            "recall": [0.5, 0.8, 0.9],
        }
    )
    assert IMPL.select_operating_threshold(results) == pytest.approx(0.3)


def test_holdout_workflow_selects_validation_then_tests_once() -> None:
    rng = np.random.default_rng(8)
    timestamps = pd.date_range("2026-01-01", periods=200, freq="h")
    y = rng.binomial(1, 0.2, 200)
    scores = np.clip(0.15 + 0.65 * y + rng.normal(0, 0.18, 200), 0, 1)
    workflow = IMPL.CostSensitiveHoldoutWorkflow(
        [0.2, 0.4, 0.6, 0.8],
        false_positive_cost=5,
        false_negative_cost=40,
    ).fit(scores, y, timestamps)
    assert workflow.selected_threshold_ == IMPL.select_operating_threshold(
        workflow.validation_results_
    )
    assert workflow.test_evaluation_count_ == 1
    assert len(workflow.test_indices_) == 30
    assert workflow.predict(np.array([0.0, 1.0])).shape == (2,)
    with pytest.raises(RuntimeError, match="fit"):
        IMPL.CostSensitiveHoldoutWorkflow(
            [0.5], false_positive_cost=1, false_negative_cost=1
        ).predict(np.array([0.5]))
