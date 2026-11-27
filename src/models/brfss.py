"""Load the trained BRFSS pipeline and run inference on one survey response."""

from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Annotated
from warnings import catch_warnings, simplefilter

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.exceptions import InconsistentVersionWarning

from ..config import model_path

Binary = Annotated[int, Field(ge=0, le=1)]
DECISION_THRESHOLD = 0.5


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    Age: int = Field(ge=1, le=13, description="Nhóm tuổi BRFSS (1 = 18–24, 13 = từ 80 tuổi)")
    BMI: float = Field(gt=0, allow_inf_nan=False, description="BMI (kg/m²)")
    HighBP: Binary
    HighChol: Binary
    GenHlth: int = Field(ge=1, le=5, description="Sức khỏe tự đánh giá (1 = tốt nhất, 5 = kém nhất)")


class ModelLoadError(RuntimeError):
    pass


def load_model() -> dict:
    return _load_bundle(model_path())


@lru_cache(maxsize=1)
def _load_bundle(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Chưa tìm thấy mô hình BRFSS tại {path}.")

    # joblib artifacts are executable when loaded. Use the project's own export.
    with catch_warnings():
        simplefilter("error", InconsistentVersionWarning)
        try:
            bundle = joblib.load(path)
        except InconsistentVersionWarning as exc:
            raise ModelLoadError(f"Cần scikit-learn=={exc.original_sklearn_version} để nạp mô hình BRFSS.") from exc

    features = bundle["features"]
    pipeline = bundle["pipeline"]
    if len(features) != len(PatientInput.model_fields) or set(features) != set(PatientInput.model_fields):
        raise ModelLoadError("File mô hình không khớp với bộ 5 trường BRFSS.")
    if list(pipeline.feature_names_in_) != list(features) or list(pipeline.classes_) != [0, 1]:
        raise ModelLoadError("Thứ tự đầu vào hoặc nhãn trong file mô hình BRFSS không hợp lệ.")
    if not 0 <= bundle["threshold"] <= 1:
        raise ModelLoadError("Ngưỡng trong file mô hình BRFSS không hợp lệ.")
    return bundle


def prepare_input(patient: PatientInput) -> pd.DataFrame:
    return pd.DataFrame([patient.model_dump()], dtype="float32")


def predict(frame: pd.DataFrame) -> dict:
    bundle = load_model()
    score = float(bundle["pipeline"].predict_proba(frame[bundle["features"]])[0, 1])
    if not isfinite(score) or not 0 <= score <= 1:
        raise ModelLoadError("Mô hình trả về điểm không hợp lệ.")
    return {
        "score": score,
        "label": "cao" if score >= DECISION_THRESHOLD else "thấp",
        "threshold": DECISION_THRESHOLD,
        "model_version": bundle["model_version"],
        "metrics": bundle["metrics"],
        "evaluation_threshold": float(bundle["threshold"]),
    }
