"""Reproducible educational classifier. Never use for clinical decisions."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ROOT, model_path

FEATURES = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree", "age"]
OPENML_TO_APP = dict(zip(["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"], FEATURES))
MISSING_ZERO = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def train(output: Path | None = None) -> dict:
    dataset = fetch_openml(data_id=37, as_frame=True, parser="auto", data_home=ROOT / "artifacts" / "openml_cache")
    frame = dataset.data.rename(columns=OPENML_TO_APP)
    if set(FEATURES) - set(frame.columns):
        raise ValueError(f"OpenML columns changed: {list(frame.columns)}")
    x = frame[FEATURES].apply(pd.to_numeric, errors="raise").copy()
    x[MISSING_ZERO] = x[MISSING_ZERO].replace(0, np.nan)
    y = dataset.target.astype(str).str.lower().map({"tested_positive": 1, "tested_negative": 0})
    if y.isna().any():
        raise ValueError("Unexpected target labels from OpenML")
    x_train, x_test, y_train, y_test = train_test_split(x, y.astype(int), test_size=0.2, random_state=42, stratify=y)
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(x_train, y_train)
    scores = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "dataset": "OpenML 37 (Pima Indians Diabetes, version 1)",
        "train_rows": len(x_train), "test_rows": len(x_test),
        "accuracy": round(float(accuracy_score(y_test, scores >= 0.5)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, scores)), 3),
        "population_limit": "Phụ nữ Pima từ 21 tuổi; không đại diện cho dân số chung.",
    }
    destination = output or model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics, "features": FEATURES}, destination)
    destination.with_suffix(".json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def load_model():
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(f"Chưa có mô hình: {path}. Chạy python -m diabetes_agents.train trước.")
    return joblib.load(path)
