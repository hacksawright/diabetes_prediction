from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from neo4j import GraphDatabase

from .agents import Coordinator, PatientInput
from .config import model_path, neo4j_config
from .rag import Question, RAGAgent

app = FastAPI(title="Diabetes Multi-Agent Demo", version="1.0.0")
coordinator = Coordinator()
rag_agent = RAGAgent()


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/health")
def health():
    cfg = neo4j_config()
    try:
        with GraphDatabase.driver(cfg["uri"], auth=cfg["auth"]) as driver:
            driver.verify_connectivity()
            with driver.session(database=cfg["database"]) as session:
                chunks = session.run("MATCH (c:Chunk) RETURN count(c) AS count").single()["count"]
        kb = {"connected": True, "chunks": chunks, "database": cfg["database"]}
    except Exception as exc:
        kb = {"connected": False, "error": str(exc)}
    return {"model_ready": model_path().exists(), "kb": kb}


@app.post("/predict")
def predict(patient: PatientInput):
    try:
        return coordinator.run(patient)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Không truy cập được Neo4j/KB: {exc}") from exc


@app.post("/rag/ask")
def ask(question: Question):
    try:
        return rag_agent.answer(question.question.strip())
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Không thể trả lời từ KB: {exc}") from exc
