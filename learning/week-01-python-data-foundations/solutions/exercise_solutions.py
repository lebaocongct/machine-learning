"""Reference implementations for exercises/starter.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def column_zscore(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    safe_std = np.where(std == 0, 1.0, std)
    standardized = (matrix - mean) / safe_std
    standardized[:, std == 0] = 0.0
    return standardized


def compute_net_revenue(
    quantity: np.ndarray,
    unit_price: np.ndarray,
    discount_pct: np.ndarray,
) -> np.ndarray:
    quantity = np.asarray(quantity, dtype=float)
    unit_price = np.asarray(unit_price, dtype=float)
    discount_pct = np.asarray(discount_pct, dtype=float)

    if np.any(quantity < 0):
        raise ValueError("quantity must be non-negative")
    if np.any(unit_price < 0):
        raise ValueError("unit_price must be non-negative")
    if np.any((discount_pct < 0) | (discount_pct > 100)):
        raise ValueError("discount_pct must be within [0, 100]")
    return quantity * unit_price * (1 - discount_pct / 100)


def profile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        non_missing = df[column].dropna()
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_pct": float(df[column].isna().mean() * 100),
                "unique_count": int(df[column].nunique(dropna=True)),
                "sample_value": None if non_missing.empty else non_missing.iloc[0],
            }
        )
    return pd.DataFrame(rows).set_index("column")


def summarize_by_category(cleaned: pd.DataFrame) -> pd.DataFrame:
    result = (
        cleaned.groupby("category", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            units=("quantity", "sum"),
            gross_revenue=("gross_revenue", "sum"),
            net_revenue=("net_revenue", "sum"),
        )
        .sort_values("net_revenue", ascending=False)
        .reset_index(drop=True)
    )
    result["avg_order_value"] = result["net_revenue"] / result["orders"]
    return result


def join_customers(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    joined = orders.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if len(joined) != len(orders):
        raise AssertionError("left join unexpectedly changed the row count")
    return joined

