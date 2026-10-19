import hashlib, json, subprocess, sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from data.generate_data import generate_dataset, save_dataset
from src.config import DATASET_PATH, ROOT, SEED, STUDENT_ID, VERIFICATION_CODE
from src.modeling import load_models
from src.validation import ValidationError, validate_identity, validate_record

def test_identity_validation():
    validate_identity(STUDENT_ID, SEED)
    with pytest.raises(ValidationError): validate_identity("wrong@ines.ac.rw", SEED)
def test_seed_validation():
    with pytest.raises(ValidationError): validate_identity(STUDENT_ID, 1)
def test_deterministic_generation_and_data_problems():
    a, b = generate_dataset(), generate_dataset()
    assert a.equals(b) and len(a) == 273 and a.duplicated().sum() == 3
    assert a.soil_moisture_pct.isna().sum() == 14 and a.fertilizer_kg.isna().sum() == 14
def test_fingerprint_reproducible(tmp_path):
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"; save_dataset(output_path=p1); save_dataset(output_path=p2)
    assert hashlib.sha256(p1.read_bytes()).hexdigest() == hashlib.sha256(p2.read_bytes()).hexdigest()
def test_record_validation():
    valid = json.loads((ROOT / "samples" / "valid.json").read_text()); assert validate_record(valid)["season"] == "A"
    with pytest.raises(ValidationError): validate_record({"rainfall_mm": 0})
def test_saved_models_and_cli():
    load_models(); result = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--json", str(ROOT / "samples" / "valid.json")], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout)["verification_code"] == VERIFICATION_CODE
def test_invalid_cli():
    result = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--json", str(ROOT / "samples" / "invalid.json")], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "VALIDATION ERROR" in result.stderr
def test_api_endpoints():
    from app import app
    client = TestClient(app); valid = json.loads((ROOT / "samples" / "valid.json").read_text())
    assert client.get("/health").status_code == 200
    assert client.get("/model-info").json()["verification_code"] == VERIFICATION_CODE
    response = client.post("/predict", json=valid); assert response.status_code == 200 and response.json()["verification_code"] == VERIFICATION_CODE
    assert client.post("/predict", json={"rainfall_mm": 700}).status_code == 422