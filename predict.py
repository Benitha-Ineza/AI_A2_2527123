"""Command-line JSON prediction interface using saved model artifacts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from src.modeling import predict_record
from src.validation import ValidationError

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--json", required=True, dest="json_path"); args = parser.parse_args()
    try:
        record = json.loads(Path(args.json_path).read_text(encoding="utf-8")); print(json.dumps(predict_record(record), indent=2)); return 0
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(f"VALIDATION ERROR: {exc}", file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())