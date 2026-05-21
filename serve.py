"""
serve.py — точка входа для Docker.
Flask раздаёт и API (/ask, /trainer/*) и статику (frontend/index.html).
Добавлена поддержка сессий, тестов из 5 вопросов и опциональной LLM.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

from retriever import Retriever
from trainer import Trainer
from session import SessionManager, ChatMessage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# ─── ИНИЦИАЛИЗАЦИЯ ────────────────────────────────────────
retriever = Retriever(
    chapter_path=os.path.join(DATA_DIR, 'chapter.md'),
    terms_path=os.path.join(DATA_DIR, 'terms.txt'),
)
trainer = Trainer(retriever)
session_manager = SessionManager()

# LLM (опционально)
llm_service = None
LLM_ENABLED = os.getenv('LLM_ENABLED', 'false').lower() == 'true'
if LLM_ENABLED:
    try:
        from llm import LLMService, LLMConfig
        llm_service = LLMService(LLMConfig(
            base_url=os.getenv('LLM_BASE_URL', 'http://llm-server:8080'),
            timeout=int(os.getenv('LLM_TIMEOUT', '30')),
            enabled=True,
        ))
        if llm_service.is_available():
            print(f'[LLM] Сервис подключён: {llm_service.config.base_url}')
        else:
            print('[LLM] Сервис недоступен, работаем без LLM')
            llm_service = None
    except ImportError:
        print('[LLM] Модуль llm не найден, работаем без LLM')
        llm_service = None
else:
    print('[LLM] LLM отключён через LLM_ENABLED')


def get_session():
    """Извлечь или создать сессию из заголовка/параметра запроса."""
    session_id = request.headers.get('X-Session-Id') or request.args.get('session_id')
    return session_manager.get_or_create(session_id)


# ─── STATIC ──────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'chunks': len(retriever.chunks),
        'sessions': session_manager.active_sessions,
        'llm_available': llm_service.is_available() if llm_service else False,
    })


# ─── API: SESSION ────────────────────────────────────────
@app.route('/api/session/init', methods=['POST'])
def session_init():
    session = get_session()
    return jsonify({
        'session_id': session.session_id,
        'created_at': session.created_at,
        'has_history': len(session.chat_history) > 0,
        'active_test': session.test_generated and not session.test_completed,
    })


@app.route('/api/session/history', methods=['GET'])
def session_history():
    session = get_session()
    return jsonify({
        'session_id': session.session_id,
        'chat_history': [
            {'role': m.role, 'content': m.content, 'timestamp': m.timestamp}
            for m in session.chat_history
        ],
        'active_test': session.test_generated and not session.test_completed,
    })


# ─── API: CHAT (Q&A) ────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('message', '').strip() or data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Вопрос не может быть пустым'}), 400

    session = get_session()

    # Сохраняем вопрос в историю
    ts = time.time()
    session.chat_history.append(ChatMessage(role='user', content=question, timestamp=ts))

    # Поиск через retriever
    results = retriever.search_with_fallback(question, min_score=0.15)
    sources = []
    was_improved = False

    if not results:
        answer_text = 'Не удалось найти релевантный фрагмент в учебнике.'
        confidence = 'low'
    else:
        best = results[0]
        answer_text = best['text'] if isinstance(best, dict) else best.chunk.text
        score = best.get('score', 0) if isinstance(best, dict) else best.score
        confidence = 'high' if score >= 0.3 else ('medium' if score >= 0.15 else 'low')

        # Если LLM доступен — улучшаем ответ
        if llm_service and confidence != 'low':
            improved = llm_service.improve_answer(answer_text, question)
            if improved and improved != answer_text:
                answer_text = improved
                was_improved = True

        # Формируем источники
        for r in results:
            if isinstance(r, dict):
                txt = r.get('text', '')[:200]
                scr = r.get('score', 0)
            else:
                txt = r.chunk.text[:200]
                scr = r.score
            sources.append({
                'text': txt + '...',
                'score': round(scr, 3),
            })

    # Сохраняем ответ в историю
    session.chat_history.append(ChatMessage(role='assistant', content=answer_text, timestamp=time.time()))

    return jsonify({
        'answer': answer_text,
        'confidence': confidence,
        'improved': was_improved,
        'sources': sources,
    })


# ─── API: TRAINER ────────────────────────────────────────
@app.route('/api/trainer/generate', methods=['GET'])
def generate_test():
    session = get_session()

    questions = trainer.generate_test()
    session.test_generated = True
    session.test_questions = questions
    session.test_answers = [None] * len(questions)
    session.test_current_index = 0
    session.test_completed = False
    session.test_started_at = time.time()

    # Если LLM доступен — улучшаем формулировки вопросов
    if llm_service:
        for i, q in enumerate(questions):
            # Извлекаем термин из context_source: "Определение термина «{term}»"
            term_name = q.context_source
            if q.context_source.startswith('Определение термина'):
                term_name = q.context_source.replace('Определение термина «', '').rstrip('»')
            context = {}
            if hasattr(retriever, 'get_context_for_llm'):
                try:
                    context = retriever.get_context_for_llm(term_name)
                except Exception:
                    context = {}
            improved = llm_service.improve_question(q.question_text, context.get('definition', ''))
            if improved and improved != q.question_text:
                questions[i].question_text = improved
                questions[i].llm_improved = True

    # Возвращаем вопросы БЕЗ correct_answer
    safe_questions = []
    for q in questions:
        safe_questions.append({
            'id': q.id,
            'type': q.type,
            'question': q.question_text,
            'options': q.options,
            'llm_improved': q.llm_improved,
        })

    return jsonify({
        'session_id': session.session_id,
        'questions': safe_questions,
        'total': len(questions),
        'generated_at': session.test_started_at,
    })


@app.route('/api/trainer/check', methods=['POST'])
def check_answer():
    data = request.get_json()
    question_id = data.get('question_id')
    answer = data.get('answer', '').strip()

    if question_id is None:
        return jsonify({'error': 'Missing question_id'}), 400

    session = get_session()

    if not session.test_generated or session.test_completed:
        return jsonify({'error': 'Нет активного теста. Вызовите /api/trainer/generate'}), 400

    if question_id < 0 or question_id >= len(session.test_questions):
        return jsonify({'error': 'Invalid question_id'}), 400

    question = session.test_questions[question_id]
    session.test_answers[question_id] = answer

    is_correct = trainer.check_answer(question, answer)

    # Проверяем, завершён ли тест
    all_answered = all(a is not None for a in session.test_answers)
    if all_answered:
        session.test_completed = True
        session.test_results = trainer.calculate_score(
            session.test_questions, session.test_answers
        )
        session.test_results.time_spent = time.time() - (session.test_started_at or time.time())

    # Следующий неотвеченный вопрос (если есть)
    next_id = None
    if not session.test_completed:
        for i, a in enumerate(session.test_answers):
            if a is None and i > question_id:
                next_id = i
                break
        if next_id is None:
            # Если не нашли после текущего, ищем с начала
            for i, a in enumerate(session.test_answers):
                if a is None:
                    next_id = i
                    break

    response = {
        'correct': is_correct,
        'question_id': question_id,
        'test_completed': session.test_completed,
        'progress': {
            'answered': sum(1 for a in session.test_answers if a is not None),
            'total': len(session.test_questions),
            'correct_so_far': sum(
                1 for i, q in enumerate(session.test_questions)
                if session.test_answers[i] is not None
                and trainer.check_answer(q, session.test_answers[i] or '')
            ),
        },
        'next_question_id': next_id,
    }

    if not is_correct and question.type == 'definition':
        # Для открытых вопросов показываем правильный ответ при ошибке
        response['correct_answer'] = question.correct_answer[:300] + '...' if len(question.correct_answer) > 300 else question.correct_answer

    if session.test_completed and session.test_results:
        response['final_score'] = {
            'score': session.test_results.score,
            'total': session.test_results.total,
            'time_spent': round(session.test_results.time_spent, 1),
        }
        response['results'] = [
            {
                'question_id': d.question_id,
                'type': d.type,
                'question': d.question_text,
                'user_answer': d.user_answer,
                'correct_answer': d.correct_answer,
                'is_correct': d.is_correct,
            }
            for d in session.test_results.details
        ]

    return jsonify(response)


@app.route('/api/trainer/complete', methods=['POST'])
def complete_test():
    session = get_session()

    if not session.test_generated:
        return jsonify({'error': 'Нет активного теста'}), 400

    # Заполняем пропущенные ответы как пустые
    for i in range(len(session.test_answers)):
        if session.test_answers[i] is None:
            session.test_answers[i] = ''

    session.test_completed = True
    session.test_results = trainer.calculate_score(
        session.test_questions, session.test_answers
    )
    session.test_results.time_spent = time.time() - (session.test_started_at or time.time())

    return jsonify({
        'score': session.test_results.score,
        'total': session.test_results.total,
        'time_spent': round(session.test_results.time_spent, 1),
        'results': [
            {
                'question_id': d.question_id,
                'type': d.type,
                'question': d.question_text,
                'user_answer': d.user_answer,
                'correct_answer': d.correct_answer,
                'is_correct': d.is_correct,
            }
            for d in session.test_results.details
        ],
    })


@app.route('/api/trainer/reset', methods=['POST'])
def reset_test():
    session = get_session()
    session.test_generated = False
    session.test_questions = []
    session.test_answers = [None] * max(len(session.test_questions), 5)
    session.test_current_index = 0
    session.test_completed = False
    session.test_started_at = None
    session.test_results = None
    return jsonify({'message': 'Тест сброшен. Вызовите /api/trainer/generate для создания нового теста.'})


# ─── API: TERMS (совместимость) ────────────────────────
@app.route('/terms')
def list_terms():
    return jsonify({'terms': retriever.terms})


@app.route('/api/terms')
def list_terms_api():
    return jsonify({'terms': retriever.terms})


# ─── API: OLD COMPAT (прежние эндпоинты) ───────────────
@app.route('/ask', methods=['POST'])
def ask_old():
    """Старый эндпоинт /ask для обратной совместимости."""
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Вопрос не может быть пустым'}), 400
    results = retriever.search(question, top_k=3)
    if not results:
        return jsonify({'answer': 'Не удалось найти релевантный фрагмент.', 'sources': []})
    best = results[0]
    return jsonify({
        'answer': best['text'],
        'score': round(best['score'], 3),
        'sources': [{'text': r['text'][:200] + '...', 'score': round(r['score'], 3)} for r in results],
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
