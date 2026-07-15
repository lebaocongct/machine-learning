"""Generate deterministic synthetic datasets for Week 3.

Run from the project root:

    python -m data.generate_datasets
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def generate_skewed_population() -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(20260718)
    n = 5_000
    segments = rng.choice(
        ["basic", "plus", "premium"], size=n, p=[0.55, 0.30, 0.15]
    )
    segment_shift = np.select(
        [segments == "plus", segments == "premium"], [0.25, 0.55], default=0.0
    )
    monthly_spend = rng.lognormal(mean=3.35 + segment_shift, sigma=0.62)
    support_calls = rng.poisson(
        np.select(
            [segments == "plus", segments == "premium"], [1.25, 1.6], default=0.9
        )
    )
    conversion_probability = _sigmoid(
        -2.15
        + 0.018 * monthly_spend
        + 0.35 * (segments == "plus")
        + 0.70 * (segments == "premium")
        - 0.08 * support_calls
    )
    converted = rng.binomial(1, conversion_probability)
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
            "segment": segments,
            "monthly_spend": monthly_spend.round(4),
            "support_calls": support_calls,
            "converted": converted,
        }
    )
    metadata = {
        "seed": 20260718,
        "rows": n,
        "population_role": "finite population for sampling experiments",
        "spend_distribution": "segment-conditioned lognormal",
        "id_column": "customer_id",
    }
    return frame, metadata


def generate_nonlinear_regression() -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(20260719)
    n = 300
    x = rng.uniform(-3.5, 3.5, size=n)
    signal = 2.0 + 1.2 * x - 0.9 * x**2 + 0.18 * x**3 + 2.0 * np.sin(1.5 * x)
    y = signal + rng.normal(0.0, 1.15, size=n)
    frame = pd.DataFrame(
        {
            "sample_id": [f"N{i:04d}" for i in range(1, n + 1)],
            "x": x.round(6),
            "y": y.round(6),
        }
    )
    frame = frame.iloc[rng.permutation(n)].reset_index(drop=True)
    metadata = {
        "seed": 20260719,
        "rows": n,
        "target": "y",
        "features": ["x"],
        "noise_std": 1.15,
        "signal": "2 + 1.2*x - 0.9*x^2 + 0.18*x^3 + 2*sin(1.5*x)",
    }
    return frame, metadata


def generate_practical_model_selection() -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(20260720)
    n = 260
    x = rng.uniform(-4.0, 4.0, size=n)
    sensor_quality = rng.uniform(0.65, 1.0, size=n)
    signal = 8.0 - 1.4 * x + 0.55 * x**2 - 0.08 * x**3 + 1.8 * np.sin(2.0 * x)
    noise_scale = 1.1 + 1.4 * (1.0 - sensor_quality)
    y = signal + rng.normal(0.0, noise_scale)
    post_event_target_proxy = y + rng.normal(0.0, 0.08, size=n)
    frame = pd.DataFrame(
        {
            "observation_id": [f"P{i:04d}" for i in range(1, n + 1)],
            "x": x.round(6),
            "sensor_quality": sensor_quality.round(6),
            "post_event_target_proxy": post_event_target_proxy.round(6),
            "y": y.round(6),
        }
    )
    frame = frame.iloc[rng.permutation(n)].reset_index(drop=True)
    metadata = {
        "seed": 20260720,
        "rows": n,
        "target": "y",
        "allowed_model_feature": "x",
        "analysis_only_feature": "sensor_quality",
        "forbidden_feature": "post_event_target_proxy",
        "forbidden_reason": "measured after the target event; direct target leakage",
        "noise": "heteroscedastic by sensor_quality",
    }
    return frame, metadata


def main() -> None:
    datasets = {
        "skewed_population.csv": generate_skewed_population(),
        "nonlinear_regression.csv": generate_nonlinear_regression(),
        "practical_model_selection.csv": generate_practical_model_selection(),
    }
    all_metadata: dict[str, object] = {}
    for filename, (frame, metadata) in datasets.items():
        frame.to_csv(ROOT / filename, index=False)
        all_metadata[filename] = metadata
        print(f"generated {filename}: {frame.shape}")
    (ROOT / "dataset_metadata.json").write_text(
        json.dumps(all_metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
