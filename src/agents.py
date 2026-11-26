"""Four explicit agents coordinated without an external LLM or API key."""
import math

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from .kb import retrieve
from .model import FEATURES, load_model


class PatientInput(BaseModel):
    pregnancies: int = Field(ge=0, le=20, description="Số lần mang thai")
    glucose: float = Field(gt=0, le=400, description="Glucose 2 giờ sau uống đường, mg/dL")
    blood_pressure: float | None = Field(default=None, gt=0, le=250, description="Huyết áp tâm trương, mmHg")
    skin_thickness: float | None = Field(default=None, gt=0, le=100, description="Độ dày nếp gấp da, mm")
    insulin: float | None = Field(default=None, gt=0, le=1000, description="Insulin 2 giờ, µU/mL")
    bmi: float = Field(gt=0, le=80, description="BMI, kg/m²")
    diabetes_pedigree: float = Field(ge=0, le=3, description="Chỉ số phả hệ tiểu đường của bộ Pima")
    age: int = Field(ge=21, le=120)

    @model_validator(mode="after")
    def valid_population(self):
        if not math.isfinite(self.glucose) or not math.isfinite(self.bmi):
            raise ValueError("Giá trị phải hữu hạn")
        return self


class IntakeAgent:
    def run(self, patient: PatientInput) -> pd.DataFrame:
        return pd.DataFrame([{feature: getattr(patient, feature) for feature in FEATURES}], columns=FEATURES)


class PredictionAgent:
    def run(self, frame: pd.DataFrame) -> dict:
        bundle = load_model()
        probability = float(bundle["pipeline"].predict_proba(frame)[0, 1])
        return {"score": round(probability, 3), "label": "cao" if probability >= 0.5 else "thấp", "threshold": 0.5, "metrics": bundle["metrics"]}


class KnowledgeAgent:
    def run(self, query: str) -> list[dict]:
        return retrieve(query)


class ExplanationAgent:
    def run(self, prediction: dict, sources: list[dict]) -> str:
        return (f"Điểm mô hình minh họa: {prediction['score']:.1%} (ngưỡng kỹ thuật 50%). "
                "Đây là điểm của mô hình học từ bộ dữ liệu Pima, không phải xác suất bệnh thực tế hay kết quả chẩn đoán. "
                "Nếu có triệu chứng hoặc chỉ số xét nghiệm bất thường, hãy trao đổi với bác sĩ về xét nghiệm glucose và HbA1c. "
                + ("Các đoạn tham khảo Vinmec được dẫn nguồn bên dưới." if sources else "KB chưa có dữ liệu phù hợp; hãy chạy lệnh nạp KB."))


class Coordinator:
    def __init__(self):
        self.intake = IntakeAgent()
        self.prediction = PredictionAgent()
        self.knowledge = KnowledgeAgent()
        self.explanation = ExplanationAgent()

    def run(self, patient: PatientInput) -> dict:
        frame = self.intake.run(patient)
        prediction = self.prediction.run(frame)
        query = "tiểu đường type 2 triệu chứng xét nghiệm glucose HbA1c nguy cơ"
        sources = self.knowledge.run(query)
        return {"prediction": prediction, "explanation": self.explanation.run(prediction, sources), "sources": sources,
                "agents": ["IntakeAgent", "PredictionAgent", "KnowledgeAgent", "ExplanationAgent"]}
