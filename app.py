"""FastAPI inference service for the saved Musanze models."""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.config import STUDENT_NAME, VERIFICATION_CODE
from src.modeling import load_models, predict_record

app = FastAPI(title="Musanze Harvest Intelligence Service", version="1.0.0")
class FarmRecord(BaseModel):
    rainfall_mm: float = Field(ge=600, le=1400); avg_temp_c: float = Field(ge=12, le=26); soil_moisture_pct: float = Field(ge=20, le=80); fertilizer_kg: float = Field(ge=0, le=250); field_size_ha: float = Field(ge=0.2, le=4.0); season: str = Field(pattern="^[AB]$"); cooperative_support: int = Field(ge=0, le=1)

@app.get("/health")
def health():
    load_models(); return {"status": "ok", "verification_code": VERIFICATION_CODE}
@app.get("/model-info")
def model_info():
    _, _, _, metadata = load_models(); return {"service": "Musanze Harvest Intelligence Service", "student": STUDENT_NAME, "verification_code": VERIFICATION_CODE, **metadata}
@app.post("/predict")
def predict(record: FarmRecord): return predict_record(record.model_dump())