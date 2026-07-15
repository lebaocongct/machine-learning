"""Reference implementation for assessment/practical-test.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from solutions.data_cleaning_solution import build_quality_report, clean_orders


ROOT = Path(__file__).resolve().parents[1]


def summarize(cleaned: pd.DataFrame, group: str) -> pd.DataFrame:
    return (
        cleaned.groupby(group, as_index=False)
        .agg(orders=("order_id", "nunique"), net_revenue=("net_revenue", "sum"))
        .sort_values("net_revenue", ascending=False)
        .reset_index(drop=True)
    )


def save_chart(category_summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(category_summary["category"], category_summary["net_revenue"])
    ax.set(
        title="Practical test: net revenue by category",
        xlabel="Category",
        ylabel="Net revenue (currency units)",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    output_dir = ROOT / "outputs" / "reference_practical"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / "data" / "practical_test_orders.csv")
    cleaned = clean_orders(raw)
    report = build_quality_report(raw, cleaned)
    category = summarize(cleaned, "category")
    region = summarize(cleaned, "region")

    assert len(cleaned) == 10
    assert cleaned["order_id"].is_unique
    assert cleaned["discount_pct"].between(0, 100).all()

    cleaned.to_csv(output_dir / "practical_cleaned_orders.csv", index=False)
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_chart(category, output_dir / "net_revenue_by_category.png")

    print(report)
    print("\nBy category:\n", category.to_string(index=False))
    print("\nBy region:\n", region.to_string(index=False))


if __name__ == "__main__":
    main()

