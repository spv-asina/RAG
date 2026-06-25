"""
session.py — Управление сессиями (одна вкладка = одна сессия).
Состояние хранится в памяти (dict). Без БД, без авторизации.
Сессии автоматически удаляются через SESSION_TTL бездействия.
"""

import uuid
import time
from typing import Dict, List, Optional
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
    # Дополнительные поля для улучшенной работы
    truth_value: Optional[str] = None  # Для true_false: "Верно" или "Неверно" (для валидации)
    display_answer: Optional[str] = None  # Ответ для показа пользователю (если отличается от correct_answer)


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
class TermProgress:
    """Прогресс по отдельному термину."""
    term: str
    asked_count: int = 0           # сколько раз спрашивали
    correct_count: int = 0         # сколько верно
    last_asked: Optional[float] = None  # когда последний раз
    question_types_used: set = field(default_factory=set)  # какие типы вопросов уже были
    
    @property
    def is_learned(self) -> bool:
        """Критерий 'изучен': хотя бы 1 правильный ответ."""
        return self.correct_count >= 1
    
    @property
    def status(self) -> str:
        """Статус термина: 'not_asked' | 'in_progress' | 'learned'."""
        if self.asked_count == 0:
            return 'not_asked'
        if self.is_learned:
            return 'learned'
        return 'in_progress'
    
    @property
    def success_rate(self) -> float:
        if self.asked_count == 0:
            return 0.0
        return self.correct_count / self.asked_count


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
    
    # ПРОГРЕСС ПО ТЕРМИНАМ (НОВОЕ)
    term_progress: Dict[str, TermProgress] = field(default_factory=dict)
    
    # ОБЩАЯ СТАТИСТИКА (НОВОЕ)
    total_questions_asked: int = 0
    total_correct: int = 0


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
    
    # ── Методы для работы с прогрессом терминов ──────────────────────────────
    
    def get_or_create_term_progress(self, session_id: str, term: str) -> TermProgress:
        """Получить или создать прогресс по термину для сессии."""
        state = self.get(session_id)
        if not state:
            # Создаём пустой прогресс, если сессии нет (крайний случай)
            return TermProgress(term=term)
        
        if term not in state.term_progress:
            state.term_progress[term] = TermProgress(term=term)
        
        return state.term_progress[term]
    
    def update_term_progress(self, session_id: str, term: str, is_correct: bool, 
                             question_type: Optional[str] = None) -> TermProgress:
        """
        Обновить прогресс по термину после ответа на вопрос.
        
        Args:
            session_id: ID сессии
            term: термин, который спрашивали
            is_correct: верно ли ответил пользователь
            question_type: тип вопроса (definition, true_false и т.д.)
        
        Returns:
            Обновлённый TermProgress
        """
        tp = self.get_or_create_term_progress(session_id, term)
        
        tp.asked_count += 1
        if is_correct:
            tp.correct_count += 1
        tp.last_asked = time.time()
        
        if question_type:
            tp.question_types_used.add(question_type)
        
        # Также обновляем общую статистику
        state = self.get(session_id)
        if state:
            state.total_questions_asked += 1
            if is_correct:
                state.total_correct += 1
        
        return tp
    
    def get_term_progress_dict(self, session_id: str) -> Dict[str, TermProgress]:
        """Получить весь прогресс по терминам для сессии."""
        state = self.get(session_id)
        if not state:
            return {}
        return state.term_progress
