"""Structured, metadata-only execution traces for the prediction workflow."""

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from threading import Lock
from time import perf_counter

from ..config import ROOT

logger = logging.getLogger(__name__)
AUDIT_PATH = ROOT / "artifacts" / "audit" / "prediction.jsonl"
_audit_lock = Lock()


class WorkflowTrace:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.started = perf_counter()
        self.steps = []

    def append(self, step: str, version: str, status: str, **metadata):
        self.steps.append({
            "request_id": self.request_id,
            "step": step,
            "version": version,
            "status": status,
            **metadata,
        })

    def run(self, step, version, work, summarize):
        started = perf_counter()
        try:
            output = work()
            payload, evidence = summarize(output)
        except Exception as exc:
            self.append(step, version, "error", payload={"error_type": type(exc).__name__}, evidence=[],
                        duration_ms=round((perf_counter() - started) * 1000, 2))
            raise
        self.append(step, version, "ok", payload=payload, evidence=evidence,
                    duration_ms=round((perf_counter() - started) * 1000, 2))
        return output


@lru_cache(maxsize=1)
def _audit_handler():
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = AuditFileHandler(AUDIT_PATH, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


class AuditFileHandler(RotatingFileHandler):
    def handleError(self, record):
        # Let the workflow trace report an unavailable audit sink.
        raise


def write_audit(trace: WorkflowTrace, status: str) -> None:
    trace.append("write_audit", "audit-v1", "ok", payload={"workflow_status": status}, evidence=[])
    record = {
        "at": datetime.now(timezone.utc).isoformat(),
        "request_id": trace.request_id,
        "status": status,
        "duration_ms": round((perf_counter() - trace.started) * 1000, 2),
        "steps": trace.steps,
        "agents": [{**step, "agent": step["step"]} for step in trace.steps],
    }
    try:
        with _audit_lock:
            _audit_handler().handle(logging.LogRecord(
                __name__, logging.INFO, "", 0, json.dumps(record, ensure_ascii=False), (), None
            ))
    except OSError:
        trace.steps[-1]["status"] = "degraded"
        logger.warning("Could not write audit event for request %s", trace.request_id, exc_info=True)
