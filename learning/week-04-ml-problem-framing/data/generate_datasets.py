"""Generate deterministic problem-framing datasets for Week 4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_scenario_cards() -> tuple[pd.DataFrame, dict[str, object]]:
    rows = [
        {
            "scenario_id": "S01",
            "title": "Support ticket routing",
            "non_ml_goal": "Route urgent tickets to a specialist queue before first response",
            "decision": "Specialist queue or standard queue",
            "prediction_unit": "One newly created ticket",
            "prediction_time": "Ticket creation",
            "proposed_target": "Escalated within 48 hours",
            "target_kind": "binary",
            "labels_available": True,
            "actionable": True,
            "latency_requirement": "under 500 ms",
        },
        {
            "scenario_id": "S02",
            "title": "Delivery duration estimate",
            "non_ml_goal": "Show customers an estimated delivery duration at checkout",
            "decision": "Display an ETA and choose a delivery promise band",
            "prediction_unit": "One checkout order",
            "prediction_time": "Checkout confirmation",
            "proposed_target": "Delivery duration in hours",
            "target_kind": "numeric",
            "labels_available": True,
            "actionable": True,
            "latency_requirement": "under 200 ms",
        },
        {
            "scenario_id": "S03",
            "title": "Customer discovery",
            "non_ml_goal": "Discover groups of customers with similar usage patterns",
            "decision": "Choose segments for qualitative research",
            "prediction_unit": "One active customer-month",
            "prediction_time": "Monthly analysis job",
            "proposed_target": "No predefined label",
            "target_kind": "clusters",
            "labels_available": False,
            "actionable": True,
            "latency_requirement": "overnight batch",
        },
        {
            "scenario_id": "S04",
            "title": "Inventory replenishment",
            "non_ml_goal": "Reduce stockouts without excessive inventory",
            "decision": "How many units to replenish for each store-SKU",
            "prediction_unit": "One store-SKU-day",
            "prediction_time": "Daily at 02:00",
            "proposed_target": "Units demanded over the next 7 days",
            "target_kind": "numeric",
            "labels_available": True,
            "actionable": True,
            "latency_requirement": "2-hour batch window",
        },
        {
            "scenario_id": "S05",
            "title": "Invoice tax calculation",
            "non_ml_goal": "Calculate tax from jurisdiction and product rules",
            "decision": "Tax amount to charge",
            "prediction_unit": "One invoice line",
            "prediction_time": "Invoice creation",
            "proposed_target": "Tax amount",
            "target_kind": "numeric",
            "labels_available": True,
            "actionable": True,
            "latency_requirement": "under 100 ms",
        },
        {
            "scenario_id": "S06",
            "title": "Product description drafting",
            "non_ml_goal": "Draft a product description from structured attributes",
            "decision": "Human editor accepts, edits, or rejects a draft",
            "prediction_unit": "One product listing",
            "prediction_time": "Catalog authoring",
            "proposed_target": "New text content",
            "target_kind": "generated_content",
            "labels_available": False,
            "actionable": True,
            "latency_requirement": "under 10 seconds",
        },
        {
            "scenario_id": "S07",
            "title": "Unused dashboard prediction",
            "non_ml_goal": "Predict a satisfaction score for reporting",
            "decision": "No product or operational action defined",
            "prediction_unit": "One customer-week",
            "prediction_time": "Weekly reporting",
            "proposed_target": "Satisfaction score",
            "target_kind": "numeric",
            "labels_available": True,
            "actionable": False,
            "latency_requirement": "overnight batch",
        },
        {
            "scenario_id": "S08",
            "title": "Search result ordering",
            "non_ml_goal": "Order results so relevant products appear first",
            "decision": "Which items occupy top result positions",
            "prediction_unit": "One query-item pair",
            "prediction_time": "Search request",
            "proposed_target": "Relevance score/order",
            "target_kind": "ordered_list",
            "labels_available": True,
            "actionable": True,
            "latency_requirement": "under 80 ms",
        },
    ]
    frame = pd.DataFrame(rows)
    metadata = {
        "rows": len(frame),
        "purpose": "problem framing and ML suitability exercises",
        "expected_task_types": {
            "S01": "classification",
            "S02": "regression",
            "S03": "clustering",
            "S04": "regression",
            "S05": "non_ml_rule_engine",
            "S06": "generation",
            "S07": "not_ready_no_action",
            "S08": "ranking",
        },
    }
    return frame, metadata


def generate_support_triage() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(20260721)
    n = 720
    created_at = pd.date_range("2026-01-01", periods=n, freq="h")
    channel = rng.choice(["email", "chat", "phone"], n, p=[0.48, 0.37, 0.15])
    customer_tier = rng.choice(["basic", "plus", "enterprise"], n, p=[0.58, 0.30, 0.12])
    message_length = rng.lognormal(5.2, 0.65, n).round().astype(int)
    previous_tickets_30d = rng.poisson(1.5, n)
    account_age_days = rng.integers(5, 2200, n)
    sentiment_score = rng.uniform(-1.0, 1.0, n)
    billing_keyword = rng.binomial(1, 0.24, n)
    outage_active = rng.binomial(1, 0.08, n)
    late_period = np.arange(n) >= int(0.70 * n)

    logit = (
        -2.85
        + 0.45 * (channel == "phone")
        + 0.40 * (customer_tier == "enterprise")
        + 0.17 * previous_tickets_30d
        - 0.95 * sentiment_score
        + 0.75 * billing_keyword
        + 1.05 * outage_active
        + 0.28 * late_period
    )
    probability = _sigmoid(logit)
    escalated = rng.binomial(1, probability)
    triage_risk_score = _sigmoid(logit + rng.normal(0.0, 0.72, n))

    resolution_hours = rng.gamma(2.2, 5.0, n) + 16.0 * escalated
    final_satisfaction = np.clip(
        4.7 - 1.4 * escalated - 0.025 * resolution_hours + rng.normal(0, 0.45, n),
        1.0,
        5.0,
    )
    manager_override = np.where(escalated == 1, rng.binomial(1, 0.75, n), 0)

    frame = pd.DataFrame(
        {
            "ticket_id": [f"T{i:05d}" for i in range(1, n + 1)],
            "created_at": created_at.astype(str),
            "channel": channel,
            "customer_tier": customer_tier,
            "message_length": message_length,
            "previous_tickets_30d": previous_tickets_30d,
            "account_age_days": account_age_days,
            "sentiment_score": sentiment_score.round(6),
            "billing_keyword": billing_keyword,
            "outage_active": outage_active,
            "triage_risk_score": triage_risk_score.round(6),
            "escalated_within_48h": escalated,
            "resolution_hours": resolution_hours.round(4),
            "final_satisfaction": final_satisfaction.round(4),
            "manager_override": manager_override,
        }
    )
    catalog = pd.DataFrame(
        [
            ("channel", "ticket_created", 0.0, True, "request channel"),
            ("customer_tier", "ticket_created", 0.0, True, "account snapshot"),
            ("message_length", "ticket_created", 0.0, True, "message metadata"),
            ("previous_tickets_30d", "ticket_created", 0.0, True, "past-only aggregate"),
            ("account_age_days", "ticket_created", 0.0, True, "account snapshot"),
            ("sentiment_score", "ticket_created", 0.05, True, "computed after message arrives"),
            ("billing_keyword", "ticket_created", 0.05, True, "computed after message arrives"),
            ("outage_active", "ticket_created", 0.0, True, "operations snapshot"),
            ("triage_risk_score", "ticket_created", 0.1, False, "existing heuristic baseline"),
            ("escalated_within_48h", "label_generation", 48.0, False, "target label"),
            ("resolution_hours", "post_outcome", 72.0, True, "not known at triage"),
            ("final_satisfaction", "post_outcome", 96.0, True, "survey after resolution"),
            ("manager_override", "post_outcome", 24.0, True, "decision after triage"),
        ],
        columns=[
            "feature_name",
            "source_stage",
            "available_offset_hours",
            "intended_for_model",
            "notes",
        ],
    )
    metadata = {
        "seed": 20260721,
        "rows": n,
        "target": "escalated_within_48h",
        "prediction_time": "ticket creation after message parsing",
        "baseline_score": "triage_risk_score",
        "false_positive_cost": 10.0,
        "false_negative_cost": 200.0,
        "split": "chronological 70/15/15",
    }
    return frame, catalog, metadata


def generate_fraud_review() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(20260722)
    n = 900
    event_time = pd.date_range("2026-03-01", periods=n, freq="5min")
    amount = rng.lognormal(3.8, 0.95, n)
    account_age_days = rng.integers(1, 2500, n)
    velocity_1h = rng.poisson(1.8, n)
    country_risk = rng.binomial(1, 0.16, n)
    device_trust = rng.beta(4.5, 1.8, n)
    card_present = rng.binomial(1, 0.32, n)
    late_period = np.arange(n) >= int(0.70 * n)
    logit = (
        -4.25
        + 0.30 * np.log1p(amount)
        + 0.24 * velocity_1h
        + 1.15 * country_risk
        - 1.10 * device_trust
        - 0.45 * card_present
        + 0.35 * late_period
    )
    probability = _sigmoid(logit)
    confirmed_fraud = rng.binomial(1, probability)
    heuristic_risk_score = _sigmoid(logit + rng.normal(0, 0.85, n))
    chargeback_amount = confirmed_fraud * amount * rng.uniform(0.7, 1.0, n)
    investigation_code = np.where(
        confirmed_fraud == 1,
        rng.choice(["fraud_card", "fraud_account"], n),
        "cleared",
    )
    analyst_notes_length = rng.poisson(25 + 45 * confirmed_fraud, n)

    frame = pd.DataFrame(
        {
            "transaction_id": [f"F{i:05d}" for i in range(1, n + 1)],
            "event_time": event_time.astype(str),
            "amount": amount.round(4),
            "account_age_days": account_age_days,
            "velocity_1h": velocity_1h,
            "country_risk": country_risk,
            "device_trust": device_trust.round(6),
            "card_present": card_present,
            "heuristic_risk_score": heuristic_risk_score.round(6),
            "confirmed_fraud": confirmed_fraud,
            "chargeback_amount": chargeback_amount.round(4),
            "investigation_code": investigation_code,
            "analyst_notes_length": analyst_notes_length,
        }
    )
    catalog = pd.DataFrame(
        [
            ("amount", "authorization", 0.0, True, "transaction request"),
            ("account_age_days", "authorization", 0.0, True, "account snapshot"),
            ("velocity_1h", "authorization", 0.0, True, "past-only aggregate"),
            ("country_risk", "authorization", 0.0, True, "country lookup"),
            ("device_trust", "authorization", 0.0, True, "device service snapshot"),
            ("card_present", "authorization", 0.0, True, "transaction request"),
            ("heuristic_risk_score", "authorization", 0.02, False, "existing baseline"),
            ("confirmed_fraud", "label_generation", 720.0, False, "target after investigation"),
            ("chargeback_amount", "post_outcome", 1440.0, True, "known after chargeback"),
            ("investigation_code", "post_outcome", 720.0, True, "analyst conclusion"),
            ("analyst_notes_length", "post_outcome", 720.0, True, "created during investigation"),
        ],
        columns=[
            "feature_name",
            "source_stage",
            "available_offset_hours",
            "intended_for_model",
            "notes",
        ],
    )
    metadata = {
        "seed": 20260722,
        "rows": n,
        "target": "confirmed_fraud",
        "prediction_time": "transaction authorization",
        "baseline_score": "heuristic_risk_score",
        "false_positive_cost": 8.0,
        "false_negative_cost": 300.0,
        "split": "chronological 70/15/15",
    }
    return frame, catalog, metadata


def main() -> None:
    scenarios, scenario_meta = generate_scenario_cards()
    support, support_catalog, support_meta = generate_support_triage()
    fraud, fraud_catalog, fraud_meta = generate_fraud_review()
    outputs = {
        "scenario_cards.csv": scenarios,
        "support_triage.csv": support,
        "support_feature_catalog.csv": support_catalog,
        "fraud_review.csv": fraud,
        "fraud_feature_catalog.csv": fraud_catalog,
    }
    for filename, frame in outputs.items():
        frame.to_csv(ROOT / filename, index=False)
        print(f"generated {filename}: {frame.shape}")
    metadata = {
        "scenario_cards.csv": scenario_meta,
        "support_triage.csv": support_meta,
        "fraud_review.csv": fraud_meta,
    }
    (ROOT / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
