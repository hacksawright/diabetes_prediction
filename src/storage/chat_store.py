"""MySQL persistence for browser-scoped chat conversations."""
import json
from contextlib import contextmanager
from uuid import UUID, uuid4

import pymysql
from pymysql.cursors import DictCursor

from ..config import mysql_config


@contextmanager
def _connection():
    connection = pymysql.connect(
        **mysql_config(),
        charset="utf8mb4",
        cursorclass=DictCursor,
        connect_timeout=5,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_conversations (
                    id CHAR(36) NOT NULL PRIMARY KEY,
                    owner_id CHAR(36) NOT NULL,
                    title VARCHAR(120) NOT NULL DEFAULT 'Cuộc trò chuyện mới',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_chat_owner_updated (owner_id, updated_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    conversation_id CHAR(36) NOT NULL,
                    role ENUM('user', 'assistant') NOT NULL,
                    content MEDIUMTEXT NOT NULL,
                    sources JSON NULL,
                    model VARCHAR(150) NULL,
                    provider VARCHAR(32) NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'ok',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_chat_messages_conversation (conversation_id, id),
                    CONSTRAINT fk_chat_messages_conversation
                        FOREIGN KEY (conversation_id) REFERENCES chat_conversations(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        yield connection
    finally:
        connection.close()


def list_conversations(owner_id: UUID) -> list[dict]:
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM chat_conversations
                WHERE owner_id = %s
                ORDER BY updated_at DESC, id DESC
                """,
                (str(owner_id),),
            )
            return cursor.fetchall()


def ping() -> None:
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()


def create_conversation(owner_id: UUID) -> dict[str, str]:
    conversation_id = uuid4()
    title = "Cuộc trò chuyện mới"
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat_conversations (id, owner_id, title)
                VALUES (%s, %s, %s)
                """,
                (str(conversation_id), str(owner_id), title),
            )
        connection.commit()
    return {"id": str(conversation_id), "title": title}


def conversation_exists(owner_id: UUID, conversation_id: UUID) -> bool:
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM chat_conversations
                WHERE id = %s AND owner_id = %s
                LIMIT 1
                """,
                (str(conversation_id), str(owner_id)),
            )
            return cursor.fetchone() is not None


def get_messages(owner_id: UUID, conversation_id: UUID) -> list[dict] | None:
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, title
                FROM chat_conversations
                WHERE id = %s AND owner_id = %s
                LIMIT 1
                """,
                (str(conversation_id), str(owner_id)),
            )
            if cursor.fetchone() is None:
                return None

            cursor.execute(
                """
                SELECT m.id, m.role, m.content, m.sources, m.model, m.provider,
                       m.status, m.created_at
                FROM chat_messages AS m
                JOIN chat_conversations AS c ON c.id = m.conversation_id
                WHERE c.id = %s AND c.owner_id = %s
                ORDER BY m.id
                """,
                (str(conversation_id), str(owner_id)),
            )
            messages = cursor.fetchall()

    for message in messages:
        raw_sources = message["sources"]
        if isinstance(raw_sources, (str, bytes, bytearray)):
            message["sources"] = json.loads(raw_sources)
        else:
            message["sources"] = raw_sources or []
    return messages


def save_exchange(
    owner_id: UUID,
    conversation_id: UUID,
    question: str,
    response: dict,
) -> bool:
    clean_title = " ".join(question.split())
    title = clean_title[:117] + "..." if len(clean_title) > 120 else clean_title
    sources = json.dumps(response.get("sources", []), ensure_ascii=False)

    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM chat_conversations
                WHERE id = %s AND owner_id = %s
                LIMIT 1
                """,
                (str(conversation_id), str(owner_id)),
            )
            if cursor.fetchone() is None:
                return False

            cursor.execute(
                """
                INSERT INTO chat_messages (conversation_id, role, content)
                VALUES (%s, 'user', %s)
                """,
                (str(conversation_id), question),
            )
            cursor.execute(
                """
                INSERT INTO chat_messages
                    (conversation_id, role, content, sources, model, provider, status)
                VALUES (%s, 'assistant', %s, %s, %s, %s, %s)
                """,
                (
                    str(conversation_id),
                    response["answer"],
                    sources,
                    response.get("model"),
                    response.get("provider"),
                    response.get("status", "ok"),
                ),
            )
            cursor.execute(
                """
                UPDATE chat_conversations
                SET title = IF(title = 'Cuộc trò chuyện mới', %s, title),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND owner_id = %s
                """,
                (title, str(conversation_id), str(owner_id)),
            )
        connection.commit()
    return True
