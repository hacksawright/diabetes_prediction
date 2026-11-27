import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def model_path() -> Path:
    value = Path(os.getenv("MODEL_PATH") or "models/brfss_web5.bin")
    return value if value.is_absolute() else ROOT / value


def neo4j_config() -> dict[str, str]:
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "auth": (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "change-this-password")),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }


def mysql_config() -> dict[str, str | int]:
    return {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "diabetes_app"),
    }
