"""Short-lived context for follow-up questions in the local demo."""
from collections import OrderedDict
from threading import Lock
from time import monotonic
from uuid import UUID, uuid4


class PredictionSessions:
    def __init__(self):
        self._sessions = OrderedDict()
        self._lock = Lock()

    def create(self, context: dict) -> UUID:
        session_id = uuid4()
        with self._lock:
            now = monotonic()
            while self._sessions:
                oldest = next(iter(self._sessions.values()))
                if oldest[0] > now and len(self._sessions) < 128:
                    break
                self._sessions.popitem(last=False)
            self._sessions[session_id] = (now + 3600, context)
        return session_id

    def get(self, session_id: UUID) -> dict | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            expires, context = session
            if expires <= monotonic():
                del self._sessions[session_id]
                return None
            return context
