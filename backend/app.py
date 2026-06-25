"""
RAG-система по Дискретной Математике (Теория графов)
Без LLM — использует TF-IDF + cosine similarity для поиска
"""

import os
import random
import uuid
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

from retriever import Retriever
from trainer import Trainer, TestValidator

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

retriever = Retriever(
    chapter_path=os.path.join(DATA_DIR, "chapter.md"),
    terms_path=os.path.join(DATA_DIR, "terms.txt"),
)
trainer = Trainer(retriever)

# ─── Хранилище тестов в памяти ─────────────────────────────
# test_store[session_id] = { test_id, questions, started_at, answers: [...] }
test_store = {}


def _get_session_id():
    """Извлекает Session-ID из заголовка запроса."""
    return request.headers.get("X-Session-Id", "default")


# ===================================================================
# НОВЫЕ API-ЭНДПОИНТЫ
# ===================================================================

@app.route("/api/session/init", methods=["POST"])
def api_session_init():
    """Инициализация сессии."""
    request.get_json(silent=True)
    sid = _get_session_id()
    return jsonify({
        "status": "ok",
        "session_id": sid,
        "server_time": time.time(),
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Чат: задать вопрос — получить ответ из учебника."""
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Вопрос не может быть пустым"}), 400

    results = retriever.search(question, top_k=3)
    if not results:
        return jsonify({"answer": "Не удалось найти релевантный фрагмент в учебнике.", "sources": []})

    best = results[0]
    answer = best["text"]
    score = best["score"]

    return jsonify({
        "answer": answer,
        "score": round(score, 3),
        "sources": [{"text": r["text"][:200] + "...", "score": round(r["score"], 3)} for r in results],
    })


@app.route("/api/trainer/generate", methods=["GET"])
def api_trainer_generate():
    """
    Сгенерировать тест из 5 вопросов.
    Query params:
      - term (optional): конкретный термин для фокуса
    """
    sid = _get_session_id()
    term = request.args.get("term")

    # Выбираем термины
    terms = retriever.terms
    if term and term in terms:
        # Ставим указанный термин первым, остальные случайные
        other_terms = [t for t in terms if t != term]
        random.shuffle(other_terms)
        selected = [term] + other_terms[:4]
    else:
        selected = random.sample(terms, min(5, len(terms)))

    # Генерируем вопросы
    questions = trainer.generate_test(selected)

    # Сериализуем в dict
    questions_data = []
    for q in questions:
        qd = {
            "id": q.id,
            "type": q.type,
            "question_text": q.question_text,
            "correct_answer": q.correct_answer,
            "context_source": q.context_source,
        }
        if q.options:
            qd["options"] = q.options
        questions_data.append(qd)

    test_id = str(uuid.uuid4())[:8]
    test_store[sid] = {
        "test_id": test_id,
        "questions": questions,
        "started_at": time.time(),
        "answers": [],
        "completed": False,
    }

    return jsonify({
        "test_id": test_id,
        "questions": questions_data,
        "total": len(questions_data),
    })


@app.route("/api/trainer/check", methods=["POST"])
def api_trainer_check():
    """
    Проверить один ответ в тесте.
    Body: { test_id, question_index, answer, question_type, correct_answer, options }
    """
    sid = _get_session_id()
    data = request.get_json(silent=True) or {}

    test = test_store.get(sid)
    if not test:
        # Пробуем найти по test_id
        for s, t in test_store.items():
            if t.get("test_id") == data.get("test_id"):
                test = t
                break

    if not test or test.get("completed"):
        # Если тест не найден или завершён — проверяем локально
        return _local_check(data)

    q_index = data.get("question_index", 0)
    answer = data.get("answer", "").strip()
    questions = test["questions"]

    if q_index >= len(questions):
        return jsonify({"error": "Неверный индекс вопроса"}), 400

    question = questions[q_index]
    is_correct = trainer.check_answer(question, answer)

    # Сохраняем ответ
    while len(test["answers"]) <= q_index:
        test["answers"].append(None)
    test["answers"][q_index] = {
        "answer": answer,
        "is_correct": is_correct,
    }

    return jsonify({
        "is_correct": is_correct,
        "correct_answer": question.correct_answer,
        "feedback": "✅ Верно!" if is_correct else "❌ Неверно",
        "question_index": q_index,
    })


def _local_check(data):
    """Проверка ответа без сохранённого теста (fallback)."""
    answer = data.get("answer", "").strip()
    correct_answer = data.get("correct_answer", "")
    q_type = data.get("question_type", "definition")
    options = data.get("options", [])

    from dataclasses import dataclass

    @dataclass
    class _Q:
        id: int
        type: str
        question_text: str
        correct_answer: str
        options: list

    q = _Q(
        id=data.get("question_id", 0),
        type=q_type,
        question_text=data.get("question_text", ""),
        correct_answer=correct_answer,
        options=options,
    )

    is_correct = TestValidator.check_answer(q, answer, retriever)
    return jsonify({
        "is_correct": is_correct,
        "correct_answer": correct_answer,
        "feedback": "✅ Верно!" if is_correct else "❌ Неверно",
        "question_index": data.get("question_index", 0),
    })


@app.route("/api/trainer/complete", methods=["POST"])
def api_trainer_complete():
    """
    Завершить тест и получить результаты.
    Body: { test_id } (опционально)
    """
    sid = _get_session_id()
    test = test_store.get(sid)

    if not test:
        return jsonify({"error": "Тест не найден. Начните новый тест."}), 404

    test["completed"] = True
    questions = test["questions"]
    answers = [a["answer"] if a else "" for a in test["answers"]]

    results = trainer.calculate_score(questions, answers)

    details = []
    for d in results.details:
        details.append({
            "question_id": d.question_id,
            "type": d.type,
            "question_text": d.question_text,
            "user_answer": d.user_answer,
            "correct_answer": d.correct_answer,
            "is_correct": d.is_correct,
        })

    # Очищаем тест из хранилища
    del test_store[sid]

    return jsonify({
        "test_id": test["test_id"],
        "score": results.score,
        "total": results.total,
        "percentage": round(results.score / results.total * 100, 1) if results.total > 0 else 0,
        "time_spent": round(results.time_spent, 2),
        "details": details,
    })


@app.route("/api/trainer/reset", methods=["POST"])
def api_trainer_reset():
    """Сбросить состояние теста для текущей сессии."""
    sid = _get_session_id()
    if sid in test_store:
        del test_store[sid]
    return jsonify({"status": "ok"})


# ===================================================================
# СТАРЫЕ ЭНДПОИНТЫ (обратная совместимость)
# ===================================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "chunks": len(retriever.chunks)})


@app.route("/ask", methods=["POST"])
def ask():
    """Режим чата: студент задаёт вопрос, получает ответ из учебника."""
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Вопрос не может быть пустым"}), 400

    results = retriever.search(question, top_k=3)
    if not results:
        return jsonify({"answer": "Не удалось найти релевантный фрагмент в учебнике.", "sources": []})

    best = results[0]
    answer = best["text"]
    score = best["score"]

    return jsonify({
        "answer": answer,
        "score": round(score, 3),
        "sources": [{"text": r["text"][:200] + "...", "score": round(r["score"], 3)} for r in results],
    })


@app.route("/trainer/question", methods=["GET"])
def get_question():
    """Тренажёр: получить случайный вопрос по терминологии."""
    term = request.args.get("term")
    q = trainer.get_question(term)
    return jsonify(q)


@app.route("/trainer/check", methods=["POST"])
def check_answer():
    """Тренажёр: проверить ответ студента."""
    data = request.get_json()
    term = data.get("term", "")
    student_answer = data.get("answer", "").strip()

    result = trainer.check(term, student_answer)
    return jsonify(result)


@app.route("/terms", methods=["GET"])
def list_terms():
    return jsonify({"terms": retriever.terms})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
