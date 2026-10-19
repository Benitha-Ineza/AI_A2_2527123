"""Identity, dataset, and inference-input validation."""
from __future__ import annotations
from typing import Any
import pandas as pd
from .config import FEATURES, TARGETS, RANGES, SEED, STUDENT_ID

class ValidationError(ValueError):
    """Raised for invalid project, dataset, or inference input."""

def validate_identity(student_id: str, seed: int) -> None:
    """Validate the student-specific command-line signature."""
    if student_id != STUDENT_ID:
        raise ValidationError(f"student-id must be {STUDENT_ID}")
    if seed != SEED:
        raise ValidationError(f"seed must be {SEED}")

def validate_dataset(df: pd.DataFrame, allow_missing: bool = True) -> dict[str, Any]:
    """Validate generator schema, source ranges, and deliberate issues."""
    required = set(FEATURES + TARGETS)
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValidationError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    for col, (low, high) in RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        if col in {"soil_moisture_pct", "fertilizer_kg"} and allow_missing:
            values = values.dropna()
        elif values.isna().any():
            raise ValidationError(f"{col} contains missing or invalid values")
        if not values.between(low, high).all():
            raise ValidationError(f"{col} has values outside [{low}, {high}]")
    if not set(df["season"].dropna().unique()).issubset({"A", "B"}):
        raise ValidationError("season must contain only A or B")
    if not set(pd.to_numeric(df["cooperative_support"], errors="coerce").dropna().unique()).issubset({0, 1}):
        raise ValidationError("cooperative_support must contain only 0 or 1")
    if (pd.to_numeric(df["yield_tons"], errors="coerce") < 0.5).any():
        raise ValidationError("yield_tons must be at least 0.50")
    if not set(pd.to_numeric(df["high_risk"], errors="coerce").dropna().unique()).issubset({0, 1}):
        raise ValidationError("high_risk must contain only 0 or 1")
    return {"rows": int(len(df)), "duplicates": int(df.duplicated().sum()), "missing": {k: int(v) for k, v in df.isna().sum().items() if v}}

def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one JSON prediction record before model inference."""
    absent = [f for f in FEATURES if f not in record]
    if absent:
        raise ValidationError(f"Missing required field(s): {', '.join(absent)}")
    cleaned: dict[str, Any] = {}
    for col, (low, high) in RANGES.items():
        try:
            value = float(record[col])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{col} must be numeric") from exc
        if not low <= value <= high:
            raise ValidationError(f"{col} must be between {low} and {high}")
        cleaned[col] = int(value) if col == "cooperative_support" else value
    season = record["season"]
    if season not in {"A", "B"}:
        raise ValidationError("season must be either 'A' or 'B'")
    cleaned["season"] = season
    return cleaned