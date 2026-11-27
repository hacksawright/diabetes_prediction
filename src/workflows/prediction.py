"""Coordinate intake, inference, review, evidence, and the prediction report."""

import logging
from uuid import uuid4

from ..graph.retrieval import RetrievalResult, search
from ..models import brfss
from .monitoring import WorkflowTrace, write_audit
from .rule_based import REVIEW_VERSION, review_prediction

logger = logging.getLogger(__name__)


def _format_explanation(rules: dict, source_count: int) -> str:
    explanation = (
        "Phần trăm là điểm tham khảo từ mô hình BRFSS. Điểm này không dùng để xác định "
        "bạn có mắc tiểu đường hay không. Bạn có thể hỏi trợ lý để hiểu kết quả và tìm hiểu về xét nghiệm."
    )
    if rules["review_required"]:
        explanation += " Lượt dự đoán này có lưu ý khi đọc kết quả."
    if not source_count:
        explanation += " Lượt dự đoán này chưa có nguồn tham khảo đính kèm."
    return explanation


class PredictionWorkflow:
    def run(self, patient: brfss.PatientInput, request_id: str | None = None) -> dict:
        trace = WorkflowTrace(request_id or str(uuid4()))
        try:
            frame = trace.run(
                "prepare_input", "brfss-input-v1", lambda: brfss.prepare_input(patient),
                lambda output: ({"fields_validated": list(output.columns)}, []),
            )
            prediction = trace.run(
                "model_inference", "brfss-inference-v1", lambda: brfss.predict(frame),
                lambda output: ({"model_version": output["model_version"]}, []),
            )
            rules = trace.run(
                "rule_review", REVIEW_VERSION, lambda: review_prediction(patient, prediction),
                lambda output: ({"review_required": output["review_required"]}, [flag["id"] for flag in output["flags"]]),
            )
            knowledge = self._lookup(trace)
            sources = knowledge.sources
            explanation = trace.run(
                "format_explanation", "report-v1", lambda: _format_explanation(rules, len(sources)),
                lambda _: ({"rules_applied": True, "source_count": len(sources)}, []),
            )
        except Exception:
            write_audit(trace, "error")
            raise

        status = "ok" if sources else "degraded"
        trace.append("assemble_report", "diabet-workflow-v1", status,
                     payload={"review_required": rules["review_required"], "source_count": len(sources)}, evidence=[])
        write_audit(trace, status)
        if knowledge.strategy == "unavailable":
            knowledge_status = "unavailable"
        else:
            knowledge_status = "ok" if sources else "insufficient"
        return {
            "request_id": trace.request_id,
            "prediction": prediction,
            "rules": rules,
            "explanation": explanation,
            "sources": sources,
            "knowledge_status": knowledge_status,
            "workflow_trace": trace.steps,
            # Keep the legacy response fields available for existing clients.
            "agents": [step["step"] for step in trace.steps],
            "agent_trace": [{**step, "agent": step["step"]} for step in trace.steps],
        }

    def _lookup(self, trace: WorkflowTrace) -> RetrievalResult:
        try:
            return trace.run(
                "lookup_references", "neo4j-retrieval-v1",
                lambda: search("tiểu đường type 2 yếu tố nguy cơ BMI huyết áp vận động", limit=4),
                lambda output: (
                    {"strategy": output.strategy, "attempts": output.attempts, "source_count": len(output.sources)},
                    [source["url"] for source in output.sources],
                ),
            )
        except Exception:
            logger.warning("Knowledge lookup unavailable for request %s", trace.request_id, exc_info=True)
            return RetrievalResult([], [], 0, "unavailable")
