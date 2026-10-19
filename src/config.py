"""Project constants and path helpers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
SAMPLES_DIR = ROOT / "samples"
DOCS_DIR = ROOT / "docs"

STUDENT_NAME = "INEZA Benitha"
REGISTRATION_NUMBER = "25/27123"
REGISTRATION_DIGITS = "2527123"
STUDENT_ID = "ug2527123@ines.ac.rw"
SEED = 7123
VERIFICATION_CODE = "AI-A2-7123"
BASE_ROWS = 240 + (SEED % 41)
DATASET_PATH = DATA_DIR / f"AI_A2_{REGISTRATION_DIGITS}.csv"

FEATURES = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha", "season", "cooperative_support"]
NUMERIC_FEATURES = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha", "cooperative_support"]
CATEGORICAL_FEATURES = ["season"]
TARGETS = ["yield_tons", "high_risk"]
RANGES = {
    "rainfall_mm": (600.0, 1400.0), "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0), "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0), "cooperative_support": (0.0, 1.0),
}

def ensure_directories() -> None:
    """Create all runtime output directories."""
    for path in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, PLOTS_DIR, SAMPLES_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)