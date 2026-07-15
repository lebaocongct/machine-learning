"""Starter functions for the independent exercises."""

from __future__ import annotations

import numpy as np
import pandas as pd


def column_zscore(matrix: np.ndarray) -> np.ndarray:
    """Standardize each column using population standard deviation."""
    raise NotImplementedError


def compute_net_revenue(
    quantity: np.ndarray,
    unit_price: np.ndarray,
    discount_pct: np.ndarray,
) -> np.ndarray:
    """Compute vectorized revenue after percentage discount."""
    raise NotImplementedError


def profile_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per column with dtype, missing and unique counts."""
    raise NotImplementedError


def summarize_by_category(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Return orders, units, gross and net revenue by category."""
    raise NotImplementedError


def join_customers(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Perform a validated left join and retain a merge indicator."""
    raise NotImplementedError

