"""Deterministically generate the student-specific synthetic farm dataset."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import BASE_ROWS, DATASET_PATH, SEED, ensure_directories
from src.validation import validate_identity, validate_dataset

def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    """Generate base rows, deliberate missing values, and exact duplicates."""
    if seed != SEED:
        raise ValueError(f"seed must be {SEED}")
    rng = np.random.default_rng(seed)
    n = BASE_ROWS
    rainfall = rng.uniform(600, 1400, n)
    temperature = rng.uniform(12, 26, n)
    moisture = rng.uniform(20, 80, n)
    fertilizer = rng.uniform(0, 250, n)
    field_size = rng.uniform(0.2, 4.0, n)
    season = rng.choice(np.array(["A", "B"]), n)
    support = rng.integers(0, 2, n)
    season_effect = np.where(season == "A", 0.50, -0.20)
    yield_tons = (0.004 * rainfall + 0.06 * moisture + 0.02 * fertilizer + 1.30 * field_size
                  - 0.25 * np.abs(temperature - 18) + 0.70 * support + season_effect
                  + rng.normal(0, 0.80, n))
    yield_tons = np.maximum(yield_tons, 0.50)
    median_yield_per_ha = np.median(yield_tons / field_size)
    df = pd.DataFrame({"rainfall_mm": rainfall, "avg_temp_c": temperature, "soil_moisture_pct": moisture,
                       "fertilizer_kg": fertilizer, "field_size_ha": field_size, "season": season,
                       "cooperative_support": support, "yield_tons": yield_tons,
                       "high_risk": ((yield_tons / field_size) < median_yield_per_ha).astype(int)})
    m = round(0.05 * n)
    df.loc[rng.choice(n, size=m, replace=False), "soil_moisture_pct"] = np.nan
    df.loc[rng.choice(n, size=m, replace=False), "fertilizer_kg"] = np.nan
    df = pd.concat([df, df.iloc[rng.choice(n, size=3, replace=False)].copy()], ignore_index=True)
    return df

def save_dataset(seed: int = SEED, output_path: Path = DATASET_PATH) -> tuple[pd.DataFrame, str]:
    """Generate, validate, write a four-decimal CSV, and return SHA-256."""
    ensure_directories(); output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(seed)
    validate_dataset(df, allow_missing=True)
    df.to_csv(output_path, index=False, float_format="%.4f")
    fingerprint = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return df, fingerprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", default="ug2527123@ines.ac.rw")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    validate_identity(args.student_id, args.seed)
    df, digest = save_dataset(args.seed)
    print({"rows": len(df), "sha256": digest})