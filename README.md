# Musanze Harvest Intelligence Service

**SWE 3513 – Artificial Intelligence | Individual Assignment 2**

| Item | Details |
| --- | --- |
| Student name | INEZA Benitha |
| Registration number | 25/27123 |
| Institutional email | [ug2527123@ines.ac.rw](mailto:ug2527123@ines.ac.rw) |
| Verification code | AI-A2-7123 |
| Seed | 7123 |
| Institution | INES Ruhengeri |
| Project location | `E:\AI_A2_2527123` |
| README location | `E:\AI_A2_2527123\README.md` |

## Project overview

This project is a reproducible machine-learning service for a Musanze agricultural cooperative. It estimates harvest yield as a regression problem, predicts whether a farm is at high production risk as a classification problem, and groups similar records using clustering. The system exposes prediction through a command-line interface and a FastAPI service for local use in environments where internet connectivity is limited or unavailable.

The dataset used in this project is synthetic and was generated deterministically from the required student seed (`7123`) and registration digits (`2527123`). It is designed for academic demonstration and reproducibility, not for real-world agronomic decision-making or financial eligibility assessment.

## Main features

- Synthetic dataset generation with student-specific seed and verification code.
- Data cleaning for deliberate missing values and duplicate rows.
- Train/test separation with leakage prevention and reproducible random states.
- Regression modeling for yield estimation using multiple models and evaluation metrics.
- Classification modeling for high-risk prediction with F1-based model selection.
- Clustering with K-from-2-to-6 silhouette scoring and descriptive cluster profiles.
- Model persistence for reusable inference in `predict.py` and `app.py`.
- JSON-based prediction CLI with validation errors for malformed or incomplete input.
- FastAPI service with health and model metadata endpoints plus prediction inference.
- Automated testing with `pytest` covering reproducibility, validation, CLI behavior, and API responses.

## Technology stack

The project was verified in a Windows PowerShell environment using Python 3.14. The actual dependencies from `requirements.txt` are:

- Python 3.14 (verified in the project environment)
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Joblib
- FastAPI
- Uvicorn
- Pytest
- httpx

`requirements.txt` content:

```text
numpy>=2.0,<3.0
pandas>=2.2,<3.0
scikit-learn>=1.5,<2.0
matplotlib>=3.9,<4.0
joblib>=1.4,<2.0
fastapi>=0.115,<1.0
uvicorn>=0.30,<1.0
pytest>=8.0,<9.0
httpx>=0.27,<1.0
```

## Project structure

```text
AI_A2_2527123/
├── AI_USE.md
├── README.md
├── app.py
├── data/
│   ├── AI_A2_2527123.csv
│   └── generate_data.py
├── docs/
│   └── AI_A2_2527123_REPORT.pdf
├── evidence/
├── models/
│   ├── classification_pipeline.joblib
│   ├── clustering_pipeline.joblib
│   ├── model_metadata.joblib
│   └── regression_pipeline.joblib
├── artifacts/
│   ├── classification_metrics.json
│   ├── cleaned_dataset.csv
│   ├── clustering_report.json
│   ├── data_fingerprint.txt
│   ├── pipeline_summary.json
│   ├── plots/
│   │   ├── clusters.png
│   │   ├── confusion_matrix.png
│   │   └── residuals.png
│   ├── regression_metrics.json
│   └── test_predictions.csv
├── predict.py
├── requirements.txt
├── run_all.py
├── samples/
│   ├── invalid.json
│   └── valid.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── modeling.py
│   └── validation.py
├── tests/
│   ├── conftest.py
│   └── test_system.py
└── .venv/
```

## Installation and setup

These commands are verified for Windows PowerShell and can be copied directly into the terminal from the project root.

```powershell
Set-Location 'E:\AI_A2_2527123'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduce the full AI pipeline

Run the verified full pipeline exactly as follows:

```powershell
python run_all.py --student-id ug2527123@ines.ac.rw --seed 7123
```

This command regenerates the dataset in `data/AI_A2_2527123.csv`, validates the generated data, cleans the deliberate missing-value and duplicate issues, fits preprocessing and model pipelines, saves the trained models in `models/`, writes evaluation artifacts in `artifacts/`, and creates the dataset fingerprint and summary files.

Verified output from the current project state:

- Status: completed
- Dataset rows: 273
- Dataset SHA-256: `759d98eee29266a8d54d7de3c50c4a24c4ea50d4d5b38bf7082ed4ba3cb3e014`
- Selected regression: `linear_regression`
- Selected classification: `logistic_regression`
- Selected clusters: `2`
- Verification code: `AI-A2-7123`

The fingerprint is written to `artifacts/data_fingerprint.txt` and the summary is stored in `artifacts/pipeline_summary.json`.

## Model training and evaluation

The project uses the following actual model setup:

### Regression

The system compares a mean baseline, a linear regression model, and a random forest regressor. The selected model is `linear_regression` because it achieved the lowest five-fold cross-validation MAE on the training data. The model is evaluated on a held-out test set using MAE, RMSE, and R².

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| mean_baseline | 2.520467 | 3.160378 | -0.021262 |
| linear_regression | 0.797943 | 1.067325 | 0.883520 |
| random_forest_regressor | 1.258737 | 1.561149 | 0.750800 |

Selected regression model: `linear_regression`.

### Classification

The classification pipeline compares a majority baseline, logistic regression, and a random forest classifier. The selected model is `logistic_regression` because it achieved the highest five-fold cross-validation F1-score on the training data. Evaluation uses precision, recall, F1-score, confusion matrix, and ROC-AUC.

| Model | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| majority_baseline | 0.000000 | 0.000000 | 0.000000 | 0.500000 |
| logistic_regression | 0.937500 | 0.882353 | 0.909091 | 0.986159 |
| random_forest_classifier | 0.909091 | 0.882353 | 0.895522 | 0.970588 |

Selected classification model: `logistic_regression`.

Confusion matrix for the selected model:

```text
[[32, 2],
 [4, 30]]
```

Interpretation: the selected classifier correctly identified 32 non-high-risk cases and 30 high-risk cases, with a small number of false positives and false negatives.

### Clustering

The clustering stage scales the feature set and evaluates K from 2 to 6 with silhouette scoring. The selected cluster count is `2` because it produced the highest silhouette score among the tested values.

| K | Silhouette score |
| --- | ---: |
| 2 | 0.157215 |
| 3 | 0.139650 |
| 4 | 0.141096 |
| 5 | 0.139147 |
| 6 | 0.149264 |

Selected clusters: `2`.

Cluster profiles from `artifacts/clustering_report.json`:

- Cluster 0: rainfall 994.5 mm, field size 2.07 ha, fertilizer 138.1 kg, support 1.0.
- Cluster 1: rainfall 985.0 mm, field size 2.05 ha, fertilizer 127.3 kg, support 0.0.

These are descriptive groupings for discussion and analysis, not causal recommendations for individual farms.

### Data leakage prevention and model selection

The pipeline follows the required design rules:

- The dataset is generated deterministically with a fixed seed.
- Missing values and duplicate records are introduced intentionally and handled in cleaning.
- Train/test splitting is performed before model fitting.
- Numeric imputation, scaling, and categorical encoding are handled inside pipelines.
- The selected regression and classification models are chosen from measured validation/test evidence rather than by preference.
- `yield_tons` and `high_risk` are not used as inputs to the classification pipeline.

## CLI prediction

The project includes a JSON-based prediction CLI. The valid sample is `samples/valid.json` and the invalid sample is `samples/invalid.json`.

```powershell
python predict.py --json samples/valid.json
python predict.py --json samples/invalid.json
```

Verified valid output from the current saved models:

```json
{
  "yield_estimate_tons": 12.2258,
  "high_risk": 0,
  "high_risk_probability": 0.2867,
  "cluster": 0,
  "verification_code": "AI-A2-7123"
}
```

Verified invalid output:

```text
VALIDATION ERROR: Missing required field(s): cooperative_support
```

The invalid request exits with a nonzero code and is rejected before inference. This is expected validation behavior for malformed or incomplete inflow data.

## FastAPI usage

Start the API with:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Available endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- Swagger UI: `http://127.0.0.1:8000/docs`

The application can be tested in the browser through Swagger UI at `http://127.0.0.1:8000/docs`. Users can submit valid farm records and inspect the returned JSON prediction payload. A malformed or incomplete request is rejected with a validation error and returns HTTP 422.

A root request to `/` is expected to return HTTP 404 because the service does not define a root route.

## API testing results

The following behaviors were verified from the current implementation:

- `GET /health` returned `{"status": "ok", "verification_code": "AI-A2-7123"}`.
- `GET /model-info` returned the service name, student details, verification code, feature list, seed, and selected model metadata.
- Swagger documentation at `/docs` loaded successfully.
- `POST /predict` with valid JSON returned a successful prediction payload.
- `POST /predict` with missing `cooperative_support` returned HTTP 422.

The valid prediction returned the same result as the CLI JSON output and included the verification code `AI-A2-7123`.

## Automated tests

Run the automated test suite with:

```powershell
python -m pytest -q
```

Verified result from the current project state:

```text
8 passed, 1 warning in 27.52s
```

The warning is a Starlette/httpx deprecation warning from the FastAPI test client and does not indicate a failing test. The test suite covers:

- Identity validation and seed rejection.
- Deterministic generation and reproducible fingerprints.
- Deliberate missing values and duplicate handling.
- Prediction record validation and malformed JSON rejection.
- Saved model loading and CLI inference path.
- Invalid CLI input handling.
- API endpoint behavior.

## Dataset fingerprint and reproducibility

The dataset is generated deterministically using the required random seed (`7123`) and the required generator formula. The project writes the final CSV with four-decimal numeric formatting and computes a SHA-256 fingerprint of the saved dataset bytes. The verified fingerprint is:

```text
759d98eee29266a8d54d7de3c50c4a24c4ea50d4d5b38bf7082ed4ba3cb3e014
```

The fingerprint is stored in `artifacts/data_fingerprint.txt` and includes the verification code. Another developer can rerun the pipeline with:

```powershell
python run_all.py --student-id ug2527123@ines.ac.rw --seed 7123
```

and compare the resulting SHA-256 to confirm that the dataset and metrics remain reproducible across reruns.

## Reports and deliverables

The project currently contains the following deliverables and evidence:

- Technical report PDF: `docs/AI_A2_2527123_REPORT.pdf`
- Dataset: `data/AI_A2_2527123.csv`
- Model artifacts: `models/`
- Evaluation metrics: `artifacts/regression_metrics.json` and `artifacts/classification_metrics.json`
- Clustering report: `artifacts/clustering_report.json`
- Evaluation plots: `artifacts/plots/`
- Test predictions: `artifacts/test_predictions.csv`
- Valid sample: `samples/valid.json`
- Invalid sample: `samples/invalid.json`
- Automated tests: `tests/test_system.py`
- Evidence folder: `evidence/`

No GitHub repository URL or final commit hash has been created yet for this project, and no demonstration video or additional unverified evidence is claimed as complete.

## Known limitations

This project is intentionally limited and designed for academic use:

- The dataset is synthetic rather than field-collected from real farms.
- The model is trained on simulated agricultural patterns and is not validated against operational farm data.
- Predictions are educational examples and should not be treated as agronomic advice.
- Clustering is descriptive and not a causal model of farm behavior.
- The service should not be used for automated approval, denial, financing, insurance, or public-service decisions.
- The results are specific to the generated dataset and the student-defined seed.

## GitHub repository and final commit

- Repository: Pending GitHub publication
- Final commit hash: To be recorded after final commit

This README intentionally does not invent a GitHub repository URL or commit hash. Because changing the README alters the working tree, the final submission commit should be recorded only after the final README and all project files are in their final state.

## Academic integrity

The dataset used in this project is synthetic and was generated according to the assignment rules. The project also includes an AI assistance disclosure file at `AI_USE.md`, which states that AI-assisted support was used for drafting, debugging, documentation structure, and technical writing.

The project does not claim that no AI tools were used. Instead, it acknowledges the assistance used and clearly separates that from the student’s responsibility to verify, understand, and explain the generated system and artifacts.

---

For the assignment verification, the project is identified by the student code `AI-A2-7123`, and the generated results are reproducible via the single-command pipeline documented above.