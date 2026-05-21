"""
session.py — Управление сессиями (одна вкладка = одна сессия).
Состояние хранится в памяти (dict). Без БД, без авторизации.
Сессии автоматически удаляются через SESSION_TTL бездействия.
"""

import uuid
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str        # "user" | "assistant"
    content: str
    timestamp: float


@dataclass
class QuestionData:
    id: int
    type: str                      # "definition" | "true_false" | "multiple_choice" | "fill_blank" | "matching"
    question_text: str
    correct_answer: str
    options: Optional[List[str]] = None
    context_source: str = ""
    llm_improved: bool = False


@dataclass
class QuestionResult:
    question_id: int
    type: str
    question_text: str
    user_answer: str
    correct_answer: str
    is_correct: bool


@dataclass
class TestResults:
    score: int
    total: int
    details: List[QuestionResult]
    started_at: float
    completed_at: float
    time_spent: float


@dataclass
class SessionState:
    session_id: str
    created_at: float
    last_active: float

    # Chat
    chat_history: List[ChatMessage] = field(default_factory=list)

    # Trainer / Test
    test_generated: bool = False
    test_questions: List[QuestionData] = field(default_factory=list)
    test_answers: List[Optional[str]] = field(default_factory=lambda: [None] * 5)
    test_current_index: int = 0
    test_started_at: Optional[float] = None
    test_completed: bool = False
    test_results: Optional[TestResults] = None


def generate_session_id() -> str:
    return str(uuid.uuid4())


class SessionManager:
    """Хранит состояние сессий в памяти с автоочисткой."""

    SESSION_TTL: int = 3600  # 1 час бездействия → очистка

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> SessionState:
        """Получить или создать сессию. Если session_id не указан — создать новый."""
        self._cleanup()

        if session_id and session_id in self._sessions:
            state = self._sessions[session_id]
            state.last_active = time.time()
            return state

        new_id = session_id if session_id else generate_session_id()
        now = time.time()
        state = SessionState(
            session_id=new_id,
            created_at=now,
            last_active=now,
        )
        self._sessions[new_id] = state
        return state

    def get(self, session_id: str) -> Optional[SessionState]:
        self._cleanup()
        state = self._sessions.get(session_id)
        if state:
            state.last_active = time.time()
        return state

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _cleanup(self) -> None:
        """Удалить просроченные сессии."""
        now = time.time()
        expired = [
            sid for sid, state in self._sessions.items()
            if now - state.last_active > self.SESSION_TTL
        ]
        for sid in expired:
            del self._sessions[sid]

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
