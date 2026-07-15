"""Generate deterministic synthetic regression datasets for Week 2.

Run from the project root:

    python -m data.generate_datasets
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def generate_single_feature() -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(20260715)
    x = np.linspace(-4.0, 12.0, 80)
    noise = rng.normal(0.0, 0.8, size=x.size)
    intercept = 4.0
    slope = 3.2
    y = intercept + slope * x + noise
    frame = pd.DataFrame(
        {
            "sample_id": [f"S{i:03d}" for i in range(1, len(x) + 1)],
            "x": x.round(6),
            "y": y.round(6),
        }
    )
    metadata = {
        "seed": 20260715,
        "rows": len(frame),
        "target": "y",
        "true_intercept": intercept,
        "true_coefficients": {"x": slope},
        "noise_std": 0.8,
    }
    return frame, metadata


def generate_housing() -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(20260716)
    n = 160
    area_sqm = rng.uniform(40.0, 180.0, size=n)
    bedrooms = rng.integers(1, 7, size=n)
    age_years = rng.integers(0, 41, size=n)
    distance_km = rng.uniform(0.5, 25.0, size=n)
    energy_score = rng.uniform(50.0, 100.0, size=n)

    intercept = 35.0
    coefficients = {
        "area_sqm": 2.4,
        "bedrooms": 18.0,
        "age_years": -0.9,
        "distance_km": -2.2,
        "energy_score": 0.8,
    }
    noise = rng.normal(0.0, 12.0, size=n)
    price_thousand = (
        intercept
        + coefficients["area_sqm"] * area_sqm
        + coefficients["bedrooms"] * bedrooms
        + coefficients["age_years"] * age_years
        + coefficients["distance_km"] * distance_km
        + coefficients["energy_score"] * energy_score
        + noise
    )

    frame = pd.DataFrame(
        {
            "property_id": [f"H{i:03d}" for i in range(1, n + 1)],
            "area_sqm": area_sqm.round(3),
            "bedrooms": bedrooms,
            "age_years": age_years,
            "distance_km": distance_km.round(3),
            "energy_score": energy_score.round(3),
            "price_thousand": price_thousand.round(3),
        }
    )
    frame = frame.iloc[rng.permutation(n)].reset_index(drop=True)
    metadata = {
        "seed": 20260716,
        "rows": len(frame),
        "target": "price_thousand",
        "features": list(coefficients),
        "true_intercept": intercept,
        "true_coefficients": coefficients,
        "noise_std": 12.0,
        "units": {
            "price_thousand": "thousand currency units",
            "area_sqm": "square meters",
            "distance_km": "kilometers",
        },
    }
    return frame, metadata


def generate_practical() -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(20260717)
    n = 90
    study_hours = rng.uniform(0.5, 15.0, size=n)
    practice_tests = rng.integers(0, 21, size=n)
    sleep_hours = rng.uniform(4.0, 9.0, size=n)
    absences = rng.integers(0, 11, size=n)

    intercept = 25.0
    coefficients = {
        "study_hours": 3.5,
        "practice_tests": 1.6,
        "sleep_hours": 2.0,
        "absences": -1.8,
    }
    noise = rng.normal(0.0, 2.5, size=n)
    score = (
        intercept
        + coefficients["study_hours"] * study_hours
        + coefficients["practice_tests"] * practice_tests
        + coefficients["sleep_hours"] * sleep_hours
        + coefficients["absences"] * absences
        + noise
    )

    frame = pd.DataFrame(
        {
            "student_id": [f"P{i:03d}" for i in range(1, n + 1)],
            "study_hours": study_hours.round(4),
            "practice_tests": practice_tests,
            "sleep_hours": sleep_hours.round(4),
            "absences": absences,
            "score": score.round(4),
        }
    )
    frame = frame.iloc[rng.permutation(n)].reset_index(drop=True)
    metadata = {
        "seed": 20260717,
        "rows": len(frame),
        "target": "score",
        "features": list(coefficients),
        "true_intercept": intercept,
        "true_coefficients": coefficients,
        "noise_std": 2.5,
    }
    return frame, metadata


def main() -> None:
    datasets = {
        "single_feature_regression.csv": generate_single_feature(),
        "housing_regression.csv": generate_housing(),
        "practical_regression.csv": generate_practical(),
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

