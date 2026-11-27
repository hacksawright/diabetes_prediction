import logging
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
import pymysql

from .agents.rag import Question
from .config import neo4j_config
from .graph.ingest import PROJECT_REFERENCE, PROJECT_URL
from .models.brfss import ModelLoadError, PatientInput, load_model
from .storage import chat_store
from .storage.chat_sessions import ChatSessionUnavailable, ChatSessions
from .storage.prediction_sessions import PredictionSessions
from .workflows.chat import ChatWorkflow
from .workflows.monitoring import WorkflowTrace, write_audit
from .workflows.prediction import PredictionWorkflow

logger = logging.getLogger(__name__)

app = FastAPI(title="Diabet", version="1.0.0")
prediction_workflow = PredictionWorkflow()
chat_workflow = ChatWorkflow()
prediction_sessions = PredictionSessions()
chat_sessions = ChatSessions()
web_root = Path(__file__).resolve().parent.parent / "web"
app.mount("/static", StaticFiles(directory=web_root / "assets"), name="static")


class ChatIdentity(BaseModel):
    client_id: UUID


class ChatMessageRequest(ChatIdentity, Question):
    pass


class FollowUpTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatQuestion(Question):
    history: list[FollowUpTurn] = Field(default_factory=list, max_length=8)
    prediction_id: UUID | None = None


class SessionChatQuestion(ChatQuestion):
    chat_session_id: UUID | None = None
    use_chat_memory: bool = False


@app.get("/")
def index():
    return FileResponse(web_root / "index.html")


@app.get(PROJECT_URL)
def brfss_reference():
    return FileResponse(PROJECT_REFERENCE, media_type="text/plain; charset=utf-8")


@app.exception_handler(RequestValidationError)
async def validate_request(request: Request, exc: RequestValidationError):
    response = await request_validation_exception_handler(request, exc)
    if request.method == "POST" and request.url.path == "/predict":
        trace = WorkflowTrace(str(uuid4()))
        trace.append("validate_input", "brfss-input-v1", "invalid",
                     payload={"error_count": len(exc.errors())}, evidence=[])
        write_audit(trace, "invalid")
        response.headers["X-Request-ID"] = trace.request_id
    return response


@app.get("/health")
def health():
    try:
        model_version = load_model()["model_version"]
    except Exception:
        logger.warning("BRFSS model unavailable", exc_info=True)
        model_version = None
    cfg = neo4j_config()
    try:
        with GraphDatabase.driver(cfg["uri"], auth=cfg["auth"]) as driver:
            driver.verify_connectivity()
            with driver.session(database=cfg["database"]) as session:
                chunks = session.run("MATCH (c:Chunk) RETURN count(c) AS count").single()["count"]
        kb = {"connected": True, "chunks": chunks, "database": cfg["database"]}
    except Exception as exc:
        kb = {"connected": False, "error": str(exc)}
    try:
        chat_store.ping()
        mysql = {"connected": True}
    except pymysql.MySQLError as exc:
        mysql = {"connected": False, "error": str(exc)}
    return {"model_ready": model_version is not None, "model_version": model_version, "kb": kb, "mysql": mysql}


@app.post("/predict")
def predict(patient: PatientInput, response: Response):
    request_id = str(uuid4())
    headers = {"X-Request-ID": request_id}
    try:
        report = prediction_workflow.run(patient, request_id)
    except (FileNotFoundError, ModelLoadError) as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers=headers) from exc
    except Exception as exc:
        logger.exception("BRFSS prediction failed for request %s", request_id)
        raise HTTPException(status_code=503, detail="Không thể chạy mô hình dự đoán.", headers=headers) from exc

    response.headers.update(headers)
    prediction_id = prediction_sessions.create({
        "patient_features": patient.model_dump(exclude_none=True),
        "prediction": {key: report["prediction"][key] for key in ("score", "model_version")},
        "explanation": report["explanation"],
        "flags": report["rules"]["flags"],
    })
    return {**report, "prediction_id": prediction_id}


@app.post("/predictions/{prediction_id}/questions")
def explain_prediction(prediction_id: UUID, request: ChatQuestion):
    context = prediction_sessions.get(prediction_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Lượt dự đoán đã hết phiên. Hãy tính lại kết quả.")
    question = request.question.strip()
    try:
        return chat_workflow.answer(
            question,
            history=[turn.model_dump() for turn in request.history],
            prediction_context=context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Chưa thể tạo giải thích. Hãy thử lại hoặc xem báo cáo dự đoán.") from exc


@app.post("/rag/ask")
def ask(question: SessionChatQuestion):
    try:
        context = prediction_sessions.get(question.prediction_id) if question.prediction_id else None
        if question.use_chat_memory or question.chat_session_id:
            return chat_sessions.exchange(
                question.chat_session_id, question.question,
                lambda message, history: chat_workflow.answer(message, history, context),
            )
        return chat_workflow.answer(
            question.question,
            history=[turn.model_dump() for turn in question.history],
            prediction_context=context,
        )
    except ChatSessionUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat response failed")
        raise HTTPException(status_code=503, detail="Mình chưa thể trả lời lúc này. Bạn thử lại nhé.") from exc


@app.get("/chat/conversations")
def conversations(client_id: UUID):
    try:
        return chat_store.list_conversations(client_id)
    except pymysql.MySQLError as exc:
        raise HTTPException(
            status_code=503,
            detail="Không kết nối được MySQL local. Hãy kiểm tra service, cổng và cấu hình MYSQL_*.",
        ) from exc


@app.post("/chat/conversations", status_code=201)
def new_conversation(identity: ChatIdentity):
    try:
        return chat_store.create_conversation(identity.client_id)
    except pymysql.MySQLError as exc:
        raise HTTPException(
            status_code=503,
            detail="Không kết nối được MySQL local. Hãy kiểm tra service, cổng và cấu hình MYSQL_*.",
        ) from exc


@app.get("/chat/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: UUID, client_id: UUID):
    try:
        messages = chat_store.get_messages(client_id, conversation_id)
    except pymysql.MySQLError as exc:
        raise HTTPException(
            status_code=503,
            detail="Không kết nối được MySQL local. Hãy kiểm tra service, cổng và cấu hình MYSQL_*.",
        ) from exc
    if messages is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy hội thoại.")
    return {"messages": messages}


@app.post("/chat/conversations/{conversation_id}/messages")
def send_chat_message(conversation_id: UUID, request: ChatMessageRequest):
    try:
        previous_messages = chat_store.get_messages(
            request.client_id,
            conversation_id,
        )
    except pymysql.MySQLError as exc:
        raise HTTPException(
            status_code=503,
            detail="Không kết nối được MySQL local. Hãy kiểm tra service, cổng và cấu hình MYSQL_*.",
        ) from exc
    if previous_messages is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy hội thoại.")

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in previous_messages[-8:]
    ]
    try:
        response = chat_workflow.answer(request.question.strip(), history=history)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Không thể trả lời từ KB: {exc}") from exc

    try:
        saved = chat_store.save_exchange(
            request.client_id,
            conversation_id,
            request.question.strip(),
            response,
        )
    except pymysql.MySQLError as exc:
        raise HTTPException(
            status_code=503,
            detail="Không lưu được hội thoại vào MySQL local. Hãy kiểm tra cấu hình MYSQL_*.",
        ) from exc
    if not saved:
        raise HTTPException(status_code=404, detail="Không tìm thấy hội thoại.")
    return response
