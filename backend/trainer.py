"""
Тренажёр — генерация тестов из 5 вопросов разных типов и валидация ответов.
Без LLM (LLM подключается опционально через колбэк).

Датаклассы:
  QuestionData     — один вопрос теста
  QuestionResult   — результат проверки одного вопроса
  TestResults      — агрегированный результат теста

Классы:
  TestGenerator    — генерация вопросов по шаблонам
  TestValidator    — проверка ответов (TF-IDF для открытых, точное совпадение для закрытых)
  Trainer          — основной интерфейс (обёртка над Generator + Validator)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import random
import re
import time


# ─── Датаклассы ────────────────────────────────────────────────────────────────


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


# ─── TestGenerator ─────────────────────────────────────────────────────────────


class TestGenerator:
    """
    Генерирует тест из 5 вопросов разных типов.
    Без LLM — только шаблоны. LLM подключается через опциональный колбэк.
    """

    DEFINITION_TEMPLATES = [
        "Что такое {term}?",
        "Дайте определение понятию «{term}».",
        "Как в теории графов определяется «{term}»?",
        "Объясните, что понимается под {term}.",
    ]

    TRUE_FALSE_TEMPLATES = [
        "Верно ли, что {statement}?",
        "Определите истинность утверждения: {statement}",
    ]

    def __init__(self, retriever, llm_service=None):
        self.retriever = retriever
        self.llm_service = llm_service

    def generate_test(self, terms: List[str]) -> List[QuestionData]:
        """
        Сгенерировать тест из 5 вопросов.
        Выбирает 5 случайных терминов из списка.
        Для каждого термина — свой тип вопроса (по порядку).
        """
        selected = random.sample(terms, min(5, len(terms)))

        generators = [
            self._generate_definition,
            self._generate_true_false,
            self._generate_multiple_choice,
            self._generate_fill_blank,
            self._generate_matching,
        ]

        questions = []
        for i, (term, gen) in enumerate(zip(selected, generators)):
            q = gen(i, term)
            if q:
                questions.append(q)

        return questions

    def _generate_definition(self, qid: int, term: str) -> Optional[QuestionData]:
        """Тип 1: Определение — открытый вопрос."""
        template = random.choice(self.DEFINITION_TEMPLATES)
        question_text = template.format(term=term.lower())

        reference = self.retriever.get_term_context(term)

        return QuestionData(
            id=qid,
            type="definition",
            question_text=question_text,
            correct_answer=reference,
            context_source=f"Определение термина «{term}»",
        )

    def _generate_true_false(self, qid: int, term: str) -> Optional[QuestionData]:
        """Тип 2: Верно/Неверно — утверждение на основе определения."""
        reference = self.retriever.get_term_context(term)

        # Извлекаем первое предложение с определением
        def_sentences = re.split(r'(?<=[.!?])\s+', reference)
        def_sent = def_sentences[0] if def_sentences else reference

        # С вероятностью 50% делаем утверждение неверным
        if random.random() < 0.5 and term.lower() in def_sent.lower():
            # Заменяем термин на похожий (из списка терминов) — неверное утверждение
            other_terms = [t for t in self.retriever.terms if t != term]
            wrong_term = random.choice(other_terms) if other_terms else "гиперграф"
            statement = re.sub(
                r'\b' + re.escape(term.lower()) + r'\b',
                wrong_term.lower(),
                def_sent,
                count=1,
                flags=re.IGNORECASE,
            )
            correct = False
        else:
            statement = def_sent
            correct = True

        question_text = random.choice(self.TRUE_FALSE_TEMPLATES).format(statement=statement[:200])

        return QuestionData(
            id=qid,
            type="true_false",
            question_text=question_text,
            correct_answer="Верно" if correct else "Неверно",
            options=["Верно", "Неверно"],
            context_source=f"Определение термина «{term}»",
        )

    def _generate_multiple_choice(self, qid: int, term: str) -> Optional[QuestionData]:
        """Тип 3: Множественный выбор — 1 правильный + дистракторы."""
        reference = self.retriever.get_term_context(term)

        # Правильный ответ — короткое определение (первые 100-150 символов)
        correct = reference[:150].strip()
        if len(correct) > 150:
            correct = correct[:correct.rfind(' ')] + '...'

        # Дистракторы: определения других терминов
        other_terms = [t for t in self.retriever.terms if t != term]
        random.shuffle(other_terms)
        distractors = []
        for t in other_terms[:3]:
            ctx = self.retriever.get_term_context(t)
            distractor = ctx[:120].strip()
            if len(distractor) > 120:
                distractor = distractor[:distractor.rfind(' ')] + '...'
            distractors.append(distractor)

        # Дополняем шаблонными дистракторами, если не хватило
        fallback_distractors = [
            "Это основное понятие теории графов.",
            "Данный термин относится к топологии графов.",
            "Это свойство графа, связанное с его связностью.",
        ]
        while len(distractors) < 3:
            distractors.append(fallback_distractors[len(distractors)])

        options = [correct] + distractors[:3]
        random.shuffle(options)

        question_text = f"Какое из следующих утверждений верно для понятия «{term}»?"

        return QuestionData(
            id=qid,
            type="multiple_choice",
            question_text=question_text,
            correct_answer=correct,
            options=options,
            context_source=f"Определение термина «{term}»",
        )

    def _generate_fill_blank(self, qid: int, term: str) -> Optional[QuestionData]:
        """Тип 4: Заполни пропуск — определение с пропущенным термином."""
        reference = self.retriever.get_term_context(term)

        # Ищем предложение с определением
        sentences = re.split(r'(?<=[.!?])\s+', reference)
        def_sentences = [s for s in sentences if term.lower() in s.lower()]

        if def_sentences:
            sent = def_sentences[0]
        else:
            sent = sentences[0] if sentences else reference

        # Заменяем термин на пропуск (первое вхождение)
        question_text = re.sub(re.escape(term), '__________', sent, count=1, flags=re.IGNORECASE)

        return QuestionData(
            id=qid,
            type="fill_blank",
            question_text=question_text,
            correct_answer=term,
            context_source=f"Определение термина «{term}»",
        )

    def _generate_matching(self, qid: int, term: str) -> Optional[QuestionData]:
        """Тип 5: Сопоставление — даём определение, спрашиваем термин."""
        reference = self.retriever.get_term_context(term)

        sentences = re.split(r'(?<=[.!?])\s+', reference)
        def_sentences = [s for s in sentences if term.lower() in s.lower()]

        if def_sentences:
            sent = def_sentences[0]
        else:
            sent = sentences[0] if sentences else reference[:200]

        question_text = f"Какой термин соответствует следующему определению?\n\n«{sent[:300]}»"

        return QuestionData(
            id=qid,
            type="matching",
            question_text=question_text,
            correct_answer=term,
            context_source=f"Определение термина «{term}»",
        )


# ─── TestValidator ─────────────────────────────────────────────────────────────


class TestValidator:
    """
    Проверяет ответы пользователя на вопросы теста.
    Для открытых вопросов (definition) — TF-IDF косинусное сходство.
    Для закрытых (true_false, multiple_choice) — точное сравнение.
    """

    @staticmethod
    def check_answer(question: QuestionData, user_answer: str, retriever=None) -> bool:
        if question.type == "definition":
            return TestValidator._check_open_answer(question.correct_answer, user_answer, retriever)
        elif question.type == "true_false":
            return user_answer.strip().lower() == question.correct_answer.lower()
        elif question.type == "multiple_choice":
            return user_answer.strip().lower() == question.correct_answer.lower()
        elif question.type == "fill_blank":
            return user_answer.strip().lower() == question.correct_answer.lower()
        elif question.type == "matching":
            return user_answer.strip().lower() == question.correct_answer.lower()
        return False

    @staticmethod
    def _check_open_answer(reference: str, answer: str, retriever) -> bool:
        """TF-IDF проверка для открытых ответов."""
        if not answer.strip():
            return False
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            vecs = retriever.vectorizer.transform([answer, reference])
            score = float(cosine_similarity(vecs[0], vecs[1])[0][0])
        except Exception:
            score = 0.0

        # Проверка ключевых слов
        ref_words = set(reference.lower().split())
        ans_words = set(answer.lower().split())
        overlap = len(ref_words & ans_words) / max(len(ref_words), 1)

        final = 0.6 * score + 0.4 * overlap
        return final >= 0.12  # мягкий порог

    @staticmethod
    def calculate_score(questions: List[QuestionData], answers: List[str], retriever) -> TestResults:
        started = time.time()
        details = []
        correct_count = 0

        for q, ans in zip(questions, answers):
            is_correct = TestValidator.check_answer(q, ans or "", retriever)
            if is_correct:
                correct_count += 1
            details.append(QuestionResult(
                question_id=q.id,
                type=q.type,
                question_text=q.question_text,
                user_answer=ans or "",
                correct_answer=q.correct_answer,
                is_correct=is_correct,
            ))

        completed = time.time()
        return TestResults(
            score=correct_count,
            total=len(questions),
            details=details,
            started_at=started,
            completed_at=completed,
            time_spent=0.0,  # вычисляется на стороне вызывающего
        )


# ─── Trainer (обёртка) ─────────────────────────────────────────────────────────


class Trainer:
    """
    Основной класс тренажёра.
    Координирует генерацию тестов и проверку ответов.
    """

    def __init__(self, retriever, llm_service=None):
        self.retriever = retriever
        self.generator = TestGenerator(retriever, llm_service)
        self.validator = TestValidator()

    def generate_test(self, terms: Optional[List[str]] = None) -> List[QuestionData]:
        if terms is None:
            terms = self.retriever.terms
        return self.generator.generate_test(terms)

    def check_answer(self, question: QuestionData, answer: str) -> bool:
        return self.validator.check_answer(question, answer, self.retriever)

    def calculate_score(self, questions: List[QuestionData], answers: List[str]) -> TestResults:
        return self.validator.calculate_score(questions, answers, self.retriever)
