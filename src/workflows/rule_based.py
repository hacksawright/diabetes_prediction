"""Auditable technical rules for the BRFSS prediction report."""

from ..models.brfss import PatientInput


REVIEW_VERSION = "brfss-review-v1"


def review_prediction(patient: PatientInput, prediction: dict) -> dict:
    flags = []
    # Observed extrema of BMI in the BRFSS web5 training split (seed 42).
    if prediction["model_version"] == "brfss_web5_mlp_v1" and not 12 <= patient.BMI <= 98:
        flags.append({
            "id": "BMI_OUTSIDE_TRAINING_RANGE",
            "message": "BMI nằm ngoài khoảng đã quan sát khi huấn luyện. Kiểm tra lại giá trị và đơn vị trước khi diễn giải kết quả.",
        })
    # A model-specific review band, chosen for the demo.
    if abs(prediction["score"] - prediction["threshold"]) <= 0.05 + 1e-12:
        flags.append({
            "id": "MODEL_SCORE_BORDERLINE",
            "message": "Mô hình cho tín hiệu chưa rõ ràng với bộ thông tin đã nhập. Chỉ nên xem kết quả để tham khảo.",
        })
    return {"version": REVIEW_VERSION, "review_required": bool(flags), "flags": flags}
