"""Public tests for the Week 1 challenge.

By default the tests target ``challenge.submission``. To validate the reference
solution, run:

    WEEK1_IMPL=solutions.data_cleaning_solution python -m pytest -q
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
IMPL = importlib.import_module(os.getenv("WEEK1_IMPL", "challenge.submission"))


@pytest.fixture()
def raw_orders() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "customer_orders_raw.csv")


@pytest.fixture()
def cleaned(raw_orders: pd.DataFrame) -> pd.DataFrame:
    return IMPL.clean_orders(raw_orders)


def test_to_snake_case() -> None:
    assert IMPL.to_snake_case(" Order ID ") == "order_id"
    assert IMPL.to_snake_case("Discount-Pct") == "discount_pct"


def test_does_not_mutate_input(raw_orders: pd.DataFrame) -> None:
    before = raw_orders.copy(deep=True)
    IMPL.clean_orders(raw_orders)
    pd.testing.assert_frame_equal(raw_orders, before)


def test_schema_and_row_count(cleaned: pd.DataFrame) -> None:
    required = {
        "order_id",
        "order_date",
        "customer_id",
        "product",
        "category",
        "quantity",
        "unit_price",
        "discount_pct",
        "region",
        "payment_method",
        "gross_revenue",
        "net_revenue",
    }
    assert required.issubset(cleaned.columns)
    assert len(cleaned) == 41


def test_keys_are_unique_and_required_values_present(cleaned: pd.DataFrame) -> None:
    assert cleaned["order_id"].is_unique
    required = ["order_id", "order_date", "customer_id", "product"]
    assert not cleaned[required].isna().any().any()


def test_numeric_contract(cleaned: pd.DataFrame) -> None:
    assert pd.api.types.is_integer_dtype(cleaned["quantity"])
    assert (cleaned["quantity"] > 0).all()
    assert (cleaned["unit_price"] > 0).all()
    assert cleaned["discount_pct"].between(0, 100).all()


def test_canonical_categories(cleaned: pd.DataFrame, raw_orders: pd.DataFrame) -> None:
    assert set(cleaned["category"]) <= {"electronics", "home", "office", "fashion"}
    assert set(cleaned["region"]) <= {"north", "central", "south"}
    assert set(cleaned["payment_method"]) <= {
        "card",
        "cash",
        "e_wallet",
        "bank_transfer",
    }

    bad_rows = pd.concat([raw_orders.iloc[[0]].copy()] * 3, ignore_index=True)
    bad_rows["Order ID"] = ["BAD_CATEGORY", "BAD_REGION", "BAD_PAYMENT"]
    bad_rows.loc[0, "Category"] = "toys"
    bad_rows.loc[1, "Region"] = "Moon"
    bad_rows.loc[2, "Payment Method"] = "crypto"
    extended = pd.concat([raw_orders, bad_rows], ignore_index=True)
    recleaned = IMPL.clean_orders(extended)
    assert not {"BAD_CATEGORY", "BAD_REGION", "BAD_PAYMENT"}.intersection(
        recleaned["order_id"]
    )


def test_revenue_features(cleaned: pd.DataFrame) -> None:
    expected_gross = cleaned["quantity"] * cleaned["unit_price"]
    expected_net = expected_gross * (1 - cleaned["discount_pct"] / 100)
    np.testing.assert_allclose(cleaned["gross_revenue"], expected_gross)
    np.testing.assert_allclose(cleaned["net_revenue"], expected_net)


def test_missing_discount_defaults_to_zero(cleaned: pd.DataFrame) -> None:
    row = cleaned.loc[cleaned["order_id"] == "O004"].iloc[0]
    assert row["discount_pct"] == pytest.approx(0.0)


def test_output_is_sorted(cleaned: pd.DataFrame) -> None:
    expected = cleaned.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(cleaned, expected)


def test_quality_report(raw_orders: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    report = IMPL.build_quality_report(raw_orders, cleaned)
    assert report == {
        "raw_rows": 52,
        "clean_rows": 41,
        "removed_rows": 11,
        "duplicate_order_ids": 1,
        "missing_cells_raw": 9,
    }


def test_missing_required_column_raises(raw_orders: pd.DataFrame) -> None:
    broken = raw_orders.drop(columns=["Product"])
    with pytest.raises(ValueError, match="product"):
        IMPL.clean_orders(broken)
