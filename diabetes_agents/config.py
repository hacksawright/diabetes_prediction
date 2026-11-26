import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def model_path() -> Path:
    value = Path(os.getenv("MODEL_PATH", "artifacts/diabetes_model.joblib"))
    return value if value.is_absolute() else ROOT / value


def neo4j_config() -> dict[str, str]:
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "auth": (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "change-this-password")),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }
