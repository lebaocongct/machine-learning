"""Starter implementation for the Week 1 data-cleaning challenge.

Do not open the solution until you have made a serious first attempt.
Run the public tests with:

    python -m pytest tests/test_data_cleaning.py -q
"""

from __future__ import annotations

import pandas as pd


def to_snake_case(name: str) -> str:
    """Convert a column name such as ``Order ID`` to ``order_id``."""
    raise NotImplementedError("TODO: implement to_snake_case")


def clean_orders(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, canonical and feature-enriched orders DataFrame.

    Contract:
    - never mutate ``raw``;
    - normalize column names and string fields;
    - parse date and numeric columns;
    - fill a missing discount with 0;
    - keep the first row for duplicate ``order_id`` values;
    - reject rows missing required fields or violating numeric/date rules;
    - canonicalize category, region and payment method;
    - add ``gross_revenue`` and ``net_revenue``;
    - sort by ``order_date`` then ``order_id`` and reset the index.
    """
    raise NotImplementedError("TODO: implement clean_orders")


def build_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, int]:
    """Return the required quality counters described in challenge/README.md."""
    raise NotImplementedError("TODO: implement build_quality_report")

