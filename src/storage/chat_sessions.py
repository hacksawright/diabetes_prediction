"""In-process chat history, isolated by an opaque session ID."""

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from uuid import UUID, uuid4

MAX_SESSIONS = 128
MAX_HISTORY_CHARS = 64_000
IDLE_SECONDS = 3600


class ChatSessionUnavailable(ValueError):
    pass


@dataclass
class ChatSession:
    touched_at: float
    history: list[dict[str, str]] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)
    active: int = 0


class ChatSessions:
    def __init__(self):
        self._sessions = OrderedDict()
        self._lock = Lock()

    def exchange(self, session_id: UUID | None, question: str, answer) -> dict:
        with self._lock:
            now = monotonic()
            for key, session in list(self._sessions.items()):
                if not session.active and now - session.touched_at >= IDLE_SECONDS:
                    del self._sessions[key]
            if session_id is None:
                if len(self._sessions) >= MAX_SESSIONS:
                    oldest = next((key for key, item in self._sessions.items() if not item.active), None)
                    if oldest is None:
                        raise ChatSessionUnavailable("Các phiên chat đang bận. Bạn thử lại sau nhé.")
                    del self._sessions[oldest]
                session_id = uuid4()
                self._sessions[session_id] = ChatSession(now)
            session = self._sessions.get(session_id)
            if session is None:
                raise ChatSessionUnavailable("Phiên chat đã hết hạn. Hãy chọn Cuộc trò chuyện mới.")
            session.active += 1
        try:
            with session.lock:
                if sum(len(turn["content"]) for turn in session.history) + len(question) > MAX_HISTORY_CHARS:
                    raise ChatSessionUnavailable("Phiên chat đã đạt giới hạn. Hãy chọn Cuộc trò chuyện mới.")
                response = answer(question, session.history)
                session.history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response["answer"]},
                ])
                return {**response, "chat_session_id": session_id}
        finally:
            with self._lock:
                session.active -= 1
                session.touched_at = monotonic()
                self._sessions.move_to_end(session_id)
