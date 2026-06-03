"""
serve.py — точка входа для Docker.
Flask раздаёт и API (/ask, /trainer/*) и статику (frontend/index.html).
Добавлена поддержка сессий, тестов из 5 вопросов и опциональной LLM.
"""

import sys
import os
import time
from typing import Optional, List, Dict, Tuple

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
    resp = send_from_directory(FRONTEND_DIR, 'index.html')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


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
        detected_term = None
    else:
        best = results[0]
        score = best.get('score', 0) if isinstance(best, dict) else best.score
        confidence = 'high' if score >= 0.3 else ('medium' if score >= 0.15 else 'low')

        # Извлекаем термин
        if isinstance(best, dict):
            detected_term = best.get('term_match') or (best.get('term') if best.get('term') in (best.get('text') or '') else None)
        else:
            detected_term = best.term_match or (best.chunk.terms[0] if best.chunk.terms else None)

        # Берём точное определение из term_index, если термин найден
        if detected_term:
            term_data = retriever.term_index.get(detected_term)
            if term_data:
                answer_text = term_data.get('text', '')
            else:
                answer_text = best['text'] if isinstance(best, dict) else best.chunk.text
        else:
            answer_text = best['text'] if isinstance(best, dict) else best.chunk.text

        # Если LLM доступен — улучшаем ответ
        if llm_service and confidence != 'low':
            improved = llm_service.improve_answer(answer_text, question)
            if improved and improved != answer_text:
                answer_text = improved
                was_improved = True

        # Формируем источники — полный текст чанка, без обрезания
        for r in results:
            if isinstance(r, dict):
                txt = r.get('text', '')
                scr = r.get('score', 0)
            else:
                txt = r.chunk.text
                scr = r.score
            sources.append({
                'text': txt,
                'score': round(scr, 3),
            })

    # Сохраняем ответ в историю
    session.chat_history.append(ChatMessage(role='assistant', content=answer_text, timestamp=time.time()))

    response = {
        'answer': answer_text,
        'confidence': confidence,
        'improved': was_improved,
        'sources': sources,
    }
    if detected_term:
        response['term'] = detected_term
    return jsonify(response)


# ─── API: TRAINER ────────────────────────────────────────

def _extract_term_from_context(context_source: str) -> Optional[str]:
    """
    Извлекает термин из context_source вида:
      - "Определение термина «{term}»"
      - "Определение термина «{term}» (вариация: ...)"
    
    Возвращает: термин или None, если не удалось извлечь.
    """
    if not context_source:
        return None
    
    # Пробуем извлечь из шаблона "Определение термина «...»"
    marker_start = 'Определение термина «'
    if context_source.startswith(marker_start):
        rest = context_source[len(marker_start):]
        # Ищем закрывающую кавычку »
        close_idx = rest.find('»')
        if close_idx > 0:
            return rest[:close_idx]
    
    # Fallback: если не удалось извлечь — попробуем найти в retriever.terms
    # (это крайний случай)
    return None


@app.route('/api/trainer/generate', methods=['GET', 'POST'])
def generate_test():
    session = get_session()

    # УЛУЧШЕНИЕ: используем progress_map для умного выбора терминов
    questions = trainer.generator.generate_test(
        retriever.terms,
        progress_map=session.term_progress,
        count=5
    )
    
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
            'question_text': q.question_text,
            'options': q.options,
            'llm_improved': q.llm_improved,
        })

    resp = jsonify({
        'session_id': session.session_id,
        'questions': safe_questions,
        'total': len(questions),
        'generated_at': session.test_started_at,
    })
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/api/trainer/check', methods=['POST'])
def check_answer():
    from trainer import TestValidator  # Импортируем локально, чтобы избежать циклов
    
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

    # УЛУЧШЕНИЕ: используем детальную проверку
    check_result = TestValidator.check_answer_with_details(
        question, answer, retriever
    )
    is_correct = check_result.is_correct

    # УЛУЧШЕНИЕ: обновляем прогресс термина (если удалось извлечь термин)
    term_name = _extract_term_from_context(question.context_source)
    if term_name:
        session_manager.update_term_progress(
            session.session_id,  # передаём ID сессии, а не объект
            term_name, 
            is_correct,
            question_type=question.type  # передаём тип вопроса для статистики
        )

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
        'is_correct': is_correct,
        'feedback': '✅ Верно!' if is_correct else '❌ Неверно',
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

    # УЛУЧШЕНИЕ: добавляем детали score для открытых вопросов
    if question.type == 'definition':
        response['score_details'] = {
            'is_open_question': True,
            'similarity_score': check_result.similarity_score,
            'overlap_score': check_result.overlap_score,
            'final_score': check_result.final_score,
            'similarity_pct': check_result.similarity_pct,
            'overlap_pct': check_result.overlap_pct,
            'final_pct': check_result.final_pct,
            'passing_threshold': check_result.passing_threshold,
            'threshold_pct': check_result.threshold_pct,
        }
    
    # УЛУЧШЕНИЕ: показываем правильный ответ для ВСЕХ типов вопросов при неверном ответе
    # (ранее было только для definition)
    # Используем display_answer если доступен, иначе correct_answer
    if not is_correct:
        answer_to_show = question.display_answer if question.display_answer is not None else question.correct_answer
        # Ограничиваем длину для длинных ответов (например, для definition или true_false)
        if len(answer_to_show) > 300:
            response['correct_answer'] = answer_to_show[:300] + '...'
        else:
            response['correct_answer'] = answer_to_show

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
                'question_text': d.question_text,
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
                'question_text': d.question_text,
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
    return jsonify({'reset': True, 'message': 'Тест сброшен. Вызовите /api/trainer/generate для создания нового теста.'})


@app.route('/api/trainer/progress', methods=['GET'])
def get_progress():
    """
    Возвращает общую статистику прогресса обучения терминов для текущей сессии.
    
    Статусы терминов (по session.py):
      - never_asked: вопросов = 0
      - always_wrong: вопросов ≥1, правильных = 0
      - in_progress: вопросов ≥1, 1 ≤ правильных < вопросов
      - learned: вопросов ≥1, правильных ≥1 (упрощённый критерий: хоть 1 правильный ответ)
    """
    session = get_session()
    
    # Получаем все термины из retriever
    all_terms = retriever.terms
    total_terms = len(all_terms)
    
    # Считаем статистику
    stats = {
        'never_asked': 0,
        'always_wrong': 0,
        'in_progress': 0,
        'learned': 0,
    }
    
    terms_progress = []
    
    for term_name in all_terms:
        # Получаем прогресс из сессии (или создаём пустой, если нет)
        progress = session.term_progress.get(term_name)
        
        # Если нет прогресса — создаём объект TermProgress для удобства
        if progress is None:
            # Создаём "виртуальный" прогресс со статусами по умолчанию
            status = 'never_asked'
            questions_asked = 0
            correct_answers = 0
        else:
            # Извлекаем из существующего объекта
            status = progress.status
            questions_asked = progress.asked_count
            correct_answers = progress.correct_count
        
        # Увеличиваем счётчики
        if status in stats:
            stats[status] += 1
        
        # Добавляем в список для подробного отображения (опционально)
        terms_progress.append({
            'term': term_name,
            'status': status,
            'questions_asked': questions_asked,
            'correct_answers': correct_answers,
        })
    
    # Считаем покрытие (сколько терминов хотя бы раз задали)
    terms_with_any_question = (
        stats['always_wrong'] + 
        stats['in_progress'] + 
        stats['learned']
    )
    coverage_pct = (
        0 if total_terms == 0 else 
        int((terms_with_any_question / total_terms) * 100)
    )
    
    # Считаем процент выученных (correct >= 1)
    learned_pct = (
        0 if total_terms == 0 else 
        int((stats['learned'] / total_terms) * 100)
    )
    
    return jsonify({
        'session_id': session.session_id,
        'summary': {
            'total_terms': total_terms,
            'never_asked': stats['never_asked'],
            'always_wrong': stats['always_wrong'],
            'in_progress': stats['in_progress'],
            'learned': stats['learned'],
        },
        'coverage': {
            'covered': terms_with_any_question,
            'total': total_terms,
            'percent': coverage_pct,
        },
        'learned': {
            'count': stats['learned'],
            'total': total_terms,
            'percent': learned_pct,
        },
        'terms_progress': terms_progress,  # Подробный список всех терминов (опционально)
    })


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
