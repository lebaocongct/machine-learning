"""Reference implementation for the Week 1 cleaning challenge."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
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
}

REGION_MAP = {
    "hanoi": "north",
    "ha noi": "north",
    "da nang": "central",
    "danang": "central",
    "hcm": "south",
    "hcmc": "south",
    "tp.hcm": "south",
    "ho chi minh": "south",
    "ho chi minh city": "south",
}

PAYMENT_MAP = {
    "card": "card",
    "cash": "cash",
    "e-wallet": "e_wallet",
    "ewallet": "e_wallet",
    "bank transfer": "bank_transfer",
}


def to_snake_case(name: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    return value.strip("_").lower()


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA)


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    # Parse year-first formats strictly before using day-first parsing for the
    # remaining human-entered values. This prevents ``2026-13-01`` from being
    # silently reinterpreted as 13 January.
    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    iso_dash = text.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    iso_slash = text.str.match(r"^\d{4}/\d{2}/\d{2}$", na=False)
    other = ~(iso_dash | iso_slash)

    parsed.loc[iso_dash] = pd.to_datetime(
        text.loc[iso_dash], format="%Y-%m-%d", errors="coerce"
    )
    parsed.loc[iso_slash] = pd.to_datetime(
        text.loc[iso_slash], format="%Y/%m/%d", errors="coerce"
    )
    parsed.loc[other] = pd.to_datetime(
        text.loc[other], format="mixed", dayfirst=True, errors="coerce"
    )
    return parsed


def clean_orders(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy(deep=True)
    df.columns = [to_snake_case(column) for column in df.columns]

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    text_columns = [
        "order_id",
        "customer_id",
        "product",
        "category",
        "region",
        "payment_method",
    ]
    for column in text_columns:
        df[column] = _clean_text(df[column])

    df["order_id"] = df["order_id"].str.upper()
    df["customer_id"] = df["customer_id"].str.upper()
    df["product"] = df["product"].str.replace(r"\s+", " ", regex=True)
    df["category"] = df["category"].str.lower()
    df["region"] = df["region"].str.lower().map(REGION_MAP)
    df["payment_method"] = df["payment_method"].str.lower().map(PAYMENT_MAP)

    df["order_date"] = _parse_mixed_dates(df["order_date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce").fillna(0.0)

    df = df.drop_duplicates(subset="order_id", keep="first")

    valid = (
        df[["order_id", "order_date", "customer_id", "product", "category"]]
        .notna()
        .all(axis=1)
        & df["category"].isin({"electronics", "home", "office", "fashion"})
        & df["quantity"].notna()
        & (df["quantity"] > 0)
        & np.isclose(df["quantity"] % 1, 0)
        & df["unit_price"].notna()
        & (df["unit_price"] > 0)
        & df["discount_pct"].between(0, 100, inclusive="both")
        & df["region"].notna()
        & df["payment_method"].notna()
    )
    df = df.loc[valid].copy()

    df["quantity"] = df["quantity"].astype("int64")
    df["gross_revenue"] = df["quantity"] * df["unit_price"]
    df["net_revenue"] = df["gross_revenue"] * (1 - df["discount_pct"] / 100)

    numeric_columns = ["unit_price", "discount_pct", "gross_revenue", "net_revenue"]
    df[numeric_columns] = df[numeric_columns].astype("float64")

    return df.sort_values(["order_date", "order_id"]).reset_index(drop=True)


def build_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, int]:
    normalized = raw.copy(deep=True)
    normalized.columns = [to_snake_case(column) for column in normalized.columns]

    duplicate_order_ids = int(
        _clean_text(normalized["order_id"]).str.upper().duplicated(keep="first").sum()
    )
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(cleaned)),
        "removed_rows": int(len(raw) - len(cleaned)),
        "duplicate_order_ids": duplicate_order_ids,
        "missing_cells_raw": int(raw.isna().sum().sum()),
    }
