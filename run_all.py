"""Reproducible one-command pipeline for SWE 3513 Assignment 2."""
from __future__ import annotations
import argparse, hashlib, json, sys
from data.generate_data import save_dataset
from src.config import ARTIFACTS_DIR, SEED, VERIFICATION_CODE, ensure_directories
from src.modeling import train_all
from src.validation import ValidationError, validate_identity

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete Musanze Harvest Intelligence Service pipeline.")
    parser.add_argument("--student-id", required=True); parser.add_argument("--seed", required=True, type=int); args = parser.parse_args()
    try:
        validate_identity(args.student_id, args.seed); ensure_directories(); df, fingerprint = save_dataset(args.seed); summary = train_all(df)
        (ARTIFACTS_DIR / "data_fingerprint.txt").write_text(f"verification_code={VERIFICATION_CODE}\nsha256={fingerprint}\n", encoding="utf-8")
        (ARTIFACTS_DIR / "pipeline_summary.json").write_text(json.dumps({"verification_code": VERIFICATION_CODE, "dataset_sha256": fingerprint, **summary}, indent=2), encoding="utf-8")
        print(json.dumps({"status": "completed", "verification_code": VERIFICATION_CODE, "dataset_rows": len(df), "dataset_sha256": fingerprint, "selected_regression": summary["regression"]["selected_model"], "selected_classification": summary["classification"]["selected_model"], "selected_clusters": summary["clustering_k"]}, indent=2)); return 0
    except (ValidationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())