"""Training, evaluation, persistence, and inference functions."""
from __future__ import annotations
import json
from pathlib import Path
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import (ARTIFACTS_DIR, CATEGORICAL_FEATURES, FEATURES, MODELS_DIR, NUMERIC_FEATURES, PLOTS_DIR, SEED,
                     VERIFICATION_CODE, ensure_directories)
from .validation import validate_dataset, validate_record

def _json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def make_supervised_preprocessor() -> ColumnTransformer:
    """Build training-only imputation, scaling, and categorical encoding."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)])

def _regression_metrics(y_true, prediction) -> dict:
    return {"mae": round(float(mean_absolute_error(y_true, prediction)), 6),
            "rmse": round(float(mean_squared_error(y_true, prediction) ** 0.5), 6),
            "r2": round(float(r2_score(y_true, prediction)), 6)}

def _classification_metrics(y_true, prediction, probability) -> dict:
    return {"accuracy": round(float(accuracy_score(y_true, prediction)), 6),
            "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 6),
            "f1": round(float(f1_score(y_true, prediction, zero_division=0)), 6),
            "roc_auc": round(float(roc_auc_score(y_true, probability)), 6),
            "confusion_matrix": confusion_matrix(y_true, prediction).tolist()}

def train_all(df: pd.DataFrame) -> dict:
    """Clean data, train/select all models, save required artifacts, and return summary."""
    ensure_directories(); validate_dataset(df, allow_missing=True)
    clean = df.drop_duplicates().reset_index(names="row_id")
    clean.to_csv(ARTIFACTS_DIR / "cleaned_dataset.csv", index=False, float_format="%.4f")
    X = clean[FEATURES]; y_reg = clean["yield_tons"]; y_cls = clean["high_risk"]
    train_idx, test_idx = train_test_split(clean.index, test_size=0.25, random_state=SEED, stratify=y_cls)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    yr_train, yr_test = y_reg.loc[train_idx], y_reg.loc[test_idx]
    yc_train, yc_test = y_cls.loc[train_idx], y_cls.loc[test_idx]
    regressors = {"mean_baseline": DummyRegressor(strategy="mean"), "linear_regression": LinearRegression(),
                  "random_forest_regressor": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=SEED, n_jobs=1)}
    reg_results, reg_models = {}, {}
    for name, estimator in regressors.items():
        pipe = Pipeline([("preprocess", make_supervised_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, yr_train); pred = pipe.predict(X_test)
        cv = -cross_val_score(pipe, X_train, yr_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=1).mean()
        reg_results[name] = {**_regression_metrics(yr_test, pred), "cv_mae": round(float(cv), 6)}; reg_models[name] = pipe
    selected_regression = min(reg_results, key=lambda k: reg_results[k]["cv_mae"])
    regression_model = reg_models[selected_regression]
    joblib.dump(regression_model, MODELS_DIR / "regression_pipeline.joblib")
    reg_payload = {"verification_code": VERIFICATION_CODE, "selected_model": selected_regression,
                   "selection_reason": "Lowest five-fold cross-validation MAE on training data.", "test_metrics": reg_results[selected_regression], "models": reg_results}
    _json_dump(ARTIFACTS_DIR / "regression_metrics.json", reg_payload)
    residuals = yr_test - regression_model.predict(X_test)
    plt.figure(figsize=(7, 4)); plt.scatter(regression_model.predict(X_test), residuals, color="#2d6a4f"); plt.axhline(0, color="black", linewidth=1); plt.xlabel("Predicted yield (tons)"); plt.ylabel("Residual (actual - predicted)"); plt.title("Regression residuals"); plt.tight_layout(); plt.savefig(PLOTS_DIR / "residuals.png", dpi=160); plt.close()
    classifiers = {"majority_baseline": DummyClassifier(strategy="most_frequent"), "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
                   "random_forest_classifier": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=1)}
    cls_results, cls_models = {}, {}
    for name, estimator in classifiers.items():
        pipe = Pipeline([("preprocess", make_supervised_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, yc_train); pred = pipe.predict(X_test); prob = pipe.predict_proba(X_test)[:, 1]
        cv = cross_val_score(pipe, X_train, yc_train, cv=5, scoring="f1", n_jobs=1).mean()
        cls_results[name] = {**_classification_metrics(yc_test, pred, prob), "cv_f1": round(float(cv), 6)}; cls_models[name] = pipe
    selected_classification = max(cls_results, key=lambda k: cls_results[k]["cv_f1"])
    classification_model = cls_models[selected_classification]
    joblib.dump(classification_model, MODELS_DIR / "classification_pipeline.joblib")
    cls_payload = {"verification_code": VERIFICATION_CODE, "selected_model": selected_classification,
                   "selection_reason": "Highest five-fold cross-validation F1-score on training data.", "test_metrics": cls_results[selected_classification], "models": cls_results}
    _json_dump(ARTIFACTS_DIR / "classification_metrics.json", cls_payload)
    ConfusionMatrixDisplay.from_predictions(yc_test, classification_model.predict(X_test), display_labels=["not high risk", "high risk"], cmap="Blues"); plt.title("Classification confusion matrix"); plt.tight_layout(); plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=160); plt.close()
    cluster_features = FEATURES
    cluster_preprocess = make_supervised_preprocessor()
    Z = cluster_preprocess.fit_transform(clean[cluster_features])
    scores, candidates = {}, {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = model.fit_predict(Z); score = silhouette_score(Z, labels)
        scores[str(k)] = round(float(score), 6); candidates[k] = model
    selected_k = max(candidates, key=lambda k: scores[str(k)])
    cluster_model = Pipeline([("preprocess", cluster_preprocess), ("model", candidates[selected_k])])
    joblib.dump(cluster_model, MODELS_DIR / "clustering_pipeline.joblib")
    labels = cluster_model.predict(clean[cluster_features])
    profiles = clean.assign(cluster=labels).groupby("cluster")[NUMERIC_FEATURES].mean().round(3).to_dict(orient="index")
    interpretations = {str(k): f"Cluster {k}: farms with average rainfall {v['rainfall_mm']:.1f} mm, field size {v['field_size_ha']:.2f} ha, and fertilizer {v['fertilizer_kg']:.1f} kg; use as a discussion profile, not an individual decision." for k, v in profiles.items()}
    _json_dump(ARTIFACTS_DIR / "clustering_report.json", {"verification_code": VERIFICATION_CODE, "silhouette_scores": scores, "selected_k": selected_k, "profiles": profiles, "interpretations": interpretations, "limitation": "Silhouette scores summarize separation only; clusters are descriptive groups from synthetic data and require agronomist review."})
    plt.figure(figsize=(7, 4)); plt.scatter(clean["rainfall_mm"], clean["soil_moisture_pct"].fillna(clean["soil_moisture_pct"].median()), c=labels, cmap="viridis", s=26); plt.xlabel("Rainfall (mm)"); plt.ylabel("Soil moisture (%)"); plt.title(f"Farm clusters (K={selected_k})"); plt.tight_layout(); plt.savefig(PLOTS_DIR / "clusters.png", dpi=160); plt.close()
    test_cluster = cluster_model.predict(X_test)
    test_pred_reg = regression_model.predict(X_test); test_pred_cls = classification_model.predict(X_test); test_prob = classification_model.predict_proba(X_test)[:, 1]
    output = clean.loc[test_idx, ["row_id"]].copy(); output["actual_yield_tons"] = yr_test.values; output["predicted_yield_tons"] = test_pred_reg; output["yield_error_tons"] = output["actual_yield_tons"] - output["predicted_yield_tons"]; output["actual_high_risk"] = yc_test.values; output["predicted_high_risk"] = test_pred_cls; output["high_risk_probability"] = test_prob; output["cluster"] = test_cluster
    output.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False, float_format="%.4f")
    joblib.dump({"features": FEATURES, "seed": SEED, "verification_code": VERIFICATION_CODE, "selected_regression": selected_regression, "selected_classification": selected_classification, "selected_k": selected_k}, MODELS_DIR / "model_metadata.joblib")
    return {"raw_rows": len(df), "clean_rows": len(clean), "duplicates_removed": int(df.duplicated().sum()), "missing_before_cleaning": {"soil_moisture_pct": int(df.soil_moisture_pct.isna().sum()), "fertilizer_kg": int(df.fertilizer_kg.isna().sum())}, "regression": reg_payload, "classification": cls_payload, "clustering_k": selected_k}

def load_models() -> tuple:
    """Load persisted models in a fresh process without retraining."""
    return (joblib.load(MODELS_DIR / "regression_pipeline.joblib"), joblib.load(MODELS_DIR / "classification_pipeline.joblib"), joblib.load(MODELS_DIR / "clustering_pipeline.joblib"), joblib.load(MODELS_DIR / "model_metadata.joblib"))

def predict_record(record: dict) -> dict:
    """Run one validated record through all persisted model pipelines."""
    cleaned = validate_record(record); frame = pd.DataFrame([cleaned]); regression, classification, clustering, metadata = load_models()
    risk = int(classification.predict(frame)[0]); probability = float(classification.predict_proba(frame)[0, 1]); cluster = int(clustering.predict(frame)[0])
    return {"yield_estimate_tons": round(float(regression.predict(frame)[0]), 4), "high_risk": risk, "high_risk_probability": round(probability, 4), "cluster": cluster, "verification_code": metadata["verification_code"]}