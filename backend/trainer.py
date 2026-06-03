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
from typing import List, Optional, Dict, Tuple
import random
import re
import time
import pymorphy3


# ─── Морфологический хелпер ─────────────────────────────────────────────────
# pymorphy3 используется для поиска терминов в любой грамматической форме
# (именительный, родительный, творительный падежи и т.д.)

_morph = pymorphy3.MorphAnalyzer()

def _get_normal_forms(word: str) -> List[str]:
    """Возвращает все возможные нормальные формы слова через pymorphy3."""
    forms = set()
    for p in _morph.parse(word):
        nf = p.normal_form
        if nf and len(nf) > 0:
            forms.add(nf)
    return list(forms)


def _term_word_normals(term: str) -> List[List[str]]:
    """
    Для каждого слова в составном термине возвращает список его нормальных форм.
    Пример: 'Стягивание подграфа' → [['стягивание'], ['подграф']]
    """
    result = []
    for w in term.lower().split():
        result.append(_get_normal_forms(w))
    return result


def _find_term_spans(term_words: List[List[str]], text_words: list, max_gap: int = 10) -> Optional[Tuple[int, int]]:
    """
    Ищет ВСЕ слова термина в text_words (с учётом морфологии).
    Поддерживает:
    - Составные термины (до 5 слов)
    - Слова в любом порядке (для русского языка: "граф называется связным" → "Связный граф")
    - Непоследовательные слова (между словами термина могут быть другие слова)
    
    term_words: [['стягивание'], ['подграф']] — нормальные формы каждого слова термина
    text_words: список (match, [normal_forms], is_content) из _tokenize_text
    max_gap: максимальное количество слов между первым и последним найденным словом термина
    
    Возвращает (start_char, end_char) первого совпадения или None.
    """
    n = len(term_words)
    if n == 0:
        return None
    
    if n == 1:
        # Для однословных терминов — точное последовательное совпадение (быстро)
        for i in range(len(text_words)):
            word_match, word_normals, _ = text_words[i]
            if set(term_words[0]) & set(word_normals):
                start = text_words[i][0].start()
                end = text_words[i][0].end()
                return (start, end)
        return None
    
    # Для составных терминов — находим ВСЕ позиции каждого слова термина в тексте
    matches = []  # matches[j] = set(token_indices) для j-го слова термина
    for j, tw in enumerate(term_words):
        positions = set()
        for i, (m, nf, _) in enumerate(text_words):
            if set(tw) & set(nf):
                positions.add(i)
        matches.append(positions)
    
    # Если хоть одно слово термина не найдено — возвращаем None
    if any(len(p) == 0 for p in matches):
        return None
    
    # Для многословных терминов: ищем окно, содержащее ВСЕ слова термина
    # Пробуем все возможные окна (от минимального до max_gap)
    for window_size in range(n, min(max_gap + 1, len(text_words) + 1)):
        for window_start in range(len(text_words) - window_size + 1):
            window_end = window_start + window_size - 1
            
            # Проверяем, содержит ли окно все слова термина
            all_found = True
            for positions in matches:
                found = any(window_start <= pos <= window_end for pos in positions)
                if not found:
                    all_found = False
                    break
            
            if all_found:
                start = text_words[window_start][0].start()
                end = text_words[window_end][0].end()
                return (start, end)
    
    return None


def _tokenize_text(text: str) -> list:
    """
    Токенизирует текст: возвращает список (regex_match, [normal_forms], is_content_word).
    """
    result = []
    for match in re.finditer(r'[а-яёА-ЯЁ]+', text):
        word = match.group()
        if len(word) < 1:
            continue
        normals = _get_normal_forms(word)
        
        # Определяем, является ли слово знаменательным (не предлог, союз, частица)
        is_content = True
        for p in _morph.parse(word):
            if p.tag.POS in {'PREP', 'CONJ', 'PRCL', 'INTJ'}:
                is_content = False
                break
        
        if not normals and not is_content:
            # Служебное слово без нормальных форм — всё равно добавляем с маркером
            result.append((match, [], False))
        else:
            result.append((match, normals, is_content))
    return result


def _replace_term_morph(term: str, replacement: str, text: str) -> str:
    """
    Заменяет термин `term` на `replacement` в тексте `text`,
    учитывая русскую морфологию (падежи, числа).
    
    Пример: term="Граф", replacement="Вершина", text="Графом G(E,V) называется..."
    → "Вершиной G(E,V) называется..."
    
    Пример: term="Стягивание подграфа", replacement="Удаление вершины"
    → "Удалением вершины G(E,V) называется..." (если исходно было в творительном падеже)
    
    Если термин не найден — возвращает исходный текст без изменений.
    """
    term_words = _term_word_normals(term)
    tokenized = _tokenize_text(text)
    
    span = _find_term_spans(term_words, tokenized)
    if span is None:
        return text
    
    start, end = span
    
    # Определяем падеж и число первого слова термина для согласования
    # (для первого слова замены подбираем падеж, для остальных — просто заменяем)
    replacement_words = replacement.split()
    new_words = []
    
    first_word_text = text[start:end].split()[0] if text[start:end].split() else ''
    
    # Получаем грамматические характеристики первого слова оригинала
    first_parsed = _morph.parse(first_word_text) if first_word_text else []
    
    for j, repl_word in enumerate(replacement_words):
        if j == 0 and first_parsed:
            # Первое слово замены — пытаемся согласовать по падежу
            try:
                # Берём первый значимый разбор первого слова из текста
                original_tag = None
                for p in first_parsed:
                    if p.tag.case and not p.tag.POS in {'PREP', 'CONJ', 'PRCL', 'INTJ'}:
                        original_tag = p.tag
                        break
                
                if original_tag:
                    case = original_tag.case
                    number = original_tag.number
                    repl_parsed = _morph.parse(repl_word)[0]
                    
                    grammemes = set()
                    if case:
                        grammemes.add(case)
                    if number:
                        grammemes.add(number)
                    
                    if grammemes:
                        inflected = repl_parsed.inflect(grammemes)
                        if inflected:
                            new_word = inflected.word
                        else:
                            new_word = repl_word
                    else:
                        new_word = repl_word
                else:
                    new_word = repl_word
            except Exception:
                new_word = repl_word
        else:
            # Остальные слова — просто в нормальной форме
            new_word = repl_word
        
        new_words.append(new_word)
    
    replacement_text = ' '.join(new_words)
    
    # Сохраняем регистр первой буквы
    if first_word_text and first_word_text[0].isupper() and replacement_text:
        replacement_text = replacement_text[0].upper() + replacement_text[1:]
    
    return text[:start] + replacement_text + text[end:]


def _find_term_morph(term: str, text: str) -> Optional[Tuple[int, int, str]]:
    """
    Находит первое вхождение термина `term` в тексте `text`,
    учитывая морфологию. Поддерживает составные термины.
    
    Возвращает (start, end, matched_word) или None.
    """
    term_words = _term_word_normals(term)
    tokenized = _tokenize_text(text)
    
    span = _find_term_spans(term_words, tokenized)
    if span is None:
        return None
    
    start, end = span
    matched_text = text[start:end]
    return (start, end, matched_text)


def _mask_term_morph(term: str, text: str, placeholder: str = '_____') -> str:
    """
    Заменяет ВСЕ вхождения термина `term` в тексте `text` на placeholder,
    учитывая морфологию (падежи, числа).
    
    Пример: term="Петля", text="называется петлёй, граф с петлями"
    → "называется _____, граф с _____"
    
    Возвращает текст со всеми заменами.
    """
    term_words = _term_word_normals(term)
    
    result = text
    while True:
        tokenized = _tokenize_text(result)
        span = _find_term_spans(term_words, tokenized)
        if span is None:
            break
        start, end = span
        result = result[:start] + placeholder + result[end:]
    
    return result


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
class AnswerCheckResult:
    """Детальный результат проверки ответа (с score для открытых вопросов)."""
    is_correct: bool
    
    # Для открытых вопросов (definition):
    similarity_score: Optional[float] = None   # TF-IDF косинусное сходство (0.0 - 1.0)
    overlap_score: Optional[float] = None      # пересечение ключевых слов (0.0 - 1.0)
    final_score: Optional[float] = None        # финальный score (0.6*similarity + 0.4*overlap)
    
    # Порог прохождения (для информации)
    passing_threshold: float = 0.12
    
    @property
    def similarity_pct(self) -> Optional[int]:
        if self.similarity_score is None:
            return None
        return int(self.similarity_score * 100)
    
    @property
    def overlap_pct(self) -> Optional[int]:
        if self.overlap_score is None:
            return None
        return int(self.overlap_score * 100)
    
    @property
    def final_pct(self) -> Optional[int]:
        if self.final_score is None:
            return None
        return int(self.final_score * 100)
    
    @property
    def threshold_pct(self) -> int:
        return int(self.passing_threshold * 100)


# ─── Вспомогательные классы для умной генерации тестов ─────────────────────


class QuestionTypeRandomizer:
    """
    Случайный порядок типов вопросов для каждого теста.
    
    Проблема старой реализации:
        ВСЕГДА один и тот же порядок:
        1. definition
        2. true_false
        3. multiple_choice
        4. fill_blank
        5. matching
    
    Решение: для каждого теста перемешивать порядок типов.
    """
    
    ALL_TYPES = ['definition', 'true_false', 'multiple_choice', 'fill_blank', 'matching']
    
    @staticmethod
    def shuffled() -> List[str]:
        """Возвращает перемешанный список из 5 типов."""
        return random.sample(QuestionTypeRandomizer.ALL_TYPES, 5)


@dataclass
class TermSelectionInfo:
    """Информация о выбранном термине для теста."""
    term: str
    bucket: str           # 'never_asked' | 'always_wrong' | 'in_progress' | 'learned'
    priority: int          # 0 = max priority, 3 = min


class SmartTermSelector:
    """
    Умный выбор терминов для теста с учётом прогресса.
    
    Приоритеты выбора (от высокого к низкому):
    1. Термины, которые НИКОГДА не спрашивали
    2. Термины, на которые ВСЕГДА ошибались (asked > 0, correct == 0)
    3. Термины с низким % правильных (но есть хотя бы 1 верный = 'in_progress')
    4. Термины «изученные» (is_learned == True)
    
    Внутри каждого бакета — случайный выбор.
    """
    
    # Приоритетные бакеты (меньше число = выше приоритет)
    BUCKET_PRIORITY = {
        'never_asked': 0,
        'always_wrong': 1,
        'in_progress': 2,
        'learned': 3,
    }
    
    @staticmethod
    def _classify_term(term: str, progress_map: Dict[str, any]) -> Tuple[str, int]:
        """
        Классифицирует термин по бакету приоритета.
        
        Returns: (bucket_name, priority_number)
        """
        tp = progress_map.get(term)
        
        if tp is None:
            # Нет прогресса → никогда не спрашивали
            return ('never_asked', 0)
        
        if tp.asked_count == 0:
            return ('never_asked', 0)
        
        if tp.correct_count == 0:
            # Спрашивали, но всегда ошибались
            return ('always_wrong', 1)
        
        if not tp.is_learned:
            # Есть ответы, но не все верные (по критерию пользователя: хотя бы 1 = изучен)
            # В нашем случае is_learned = correct >= 1, так что это условие почти никогда не сработает
            # Оставлю для совместимости на будущее
            return ('in_progress', 2)
        
        # Изучен
        return ('learned', 3)
    
    @staticmethod
    def select(all_terms: List[str], 
               progress_map: Dict[str, any],  # Dict[str, TermProgress] или пустой dict
               count: int = 5) -> List[TermSelectionInfo]:
        """
        Выбирает count терминов с учётом прогресса.
        
        Args:
            all_terms: все доступные термины
            progress_map: словарь TermProgress по терминам (может быть пустым)
            count: сколько терминов выбрать
        
        Returns:
            Список TermSelectionInfo (термин + информация о бакете)
        """
        
        # 1. Классифицируем все термины по бакетам
        buckets = {
            'never_asked': [],
            'always_wrong': [],
            'in_progress': [],
            'learned': [],
        }
        
        for term in all_terms:
            bucket, priority = SmartTermSelector._classify_term(term, progress_map)
            buckets[bucket].append(term)
        
        # 2. Перемешиваем каждый бакет (случайный выбор внутри бакета)
        for bucket_name in buckets:
            random.shuffle(buckets[bucket_name])
        
        # 3. Забираем термины в порядке приоритета
        selected = []
        priority_order = ['never_asked', 'always_wrong', 'in_progress', 'learned']
        
        for bucket_name in priority_order:
            while buckets[bucket_name] and len(selected) < count:
                term = buckets[bucket_name].pop()
                selected.append(TermSelectionInfo(
                    term=term,
                    bucket=bucket_name,
                    priority=SmartTermSelector.BUCKET_PRIORITY[bucket_name],
                ))
        
        # 4. Если не хватило (все в learned) — добавляем из learned
        # (повторяем, если все изучены)
        while len(selected) < count and buckets['learned']:
            term = buckets['learned'].pop()
            selected.append(TermSelectionInfo(
                term=term,
                bucket='learned',
                priority=3,
            ))
        
        # 5. Финальный safety: если даже так не хватило — берём из исходного списка случайные
        # (крайний случай, если count > len(all_terms))
        remaining_needed = count - len(selected)
        if remaining_needed > 0:
            available = [t for t in all_terms if t not in [s.term for s in selected]]
            random.shuffle(available)
            for i in range(min(remaining_needed, len(available))):
                selected.append(TermSelectionInfo(
                    term=available[i],
                    bucket='fallback',
                    priority=4,
                ))
        
        return selected


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
        
        # Маппинг типа вопроса к функции-генератору
        self._type_to_gen = {
            'definition': self._generate_definition,
            'true_false': self._generate_true_false,
            'multiple_choice': self._generate_multiple_choice,
            'fill_blank': self._generate_fill_blank,
            'matching': self._generate_matching,
        }

    def generate_test(self, terms: List[str], 
                      progress_map: Optional[Dict[str, any]] = None,
                      count: int = 5) -> List[QuestionData]:
        """
        Генерирует тест из count вопросов.
        
        УЛУЧШЕНИЯ:
        1. Случайный порядок типов вопросов (QuestionTypeRandomizer)
        2. Умный выбор терминов с учётом прогресса (SmartTermSelector)
        3. ГАРАНТИЯ: возвращает ровно count вопросов (fallback на definition)
        
        Args:
            terms: все доступные термины
            progress_map: словарь TermProgress по терминам (опционально, для умного выбора)
            count: сколько вопросов в тесте
        
        Returns:
            Список QuestionData (ровно count элементов)
        """
        # 1. Выбираем термины (с учётом прогресса, если передан)
        if progress_map:
            selected_info = SmartTermSelector.select(terms, progress_map, min(count, len(terms)))
            selected_terms = [si.term for si in selected_info]
        else:
            # Fallback: старый простой random.sample
            selected_terms = random.sample(terms, min(count, len(terms)))
        
        # 2. Случайный порядок типов вопросов!
        shuffled_types = QuestionTypeRandomizer.shuffled()
        
        questions = []
        
        # УЛУЧШЕНИЕ: гарантируем, что получим ровно столько вопросов, сколько нужно
        for i, (term, q_type) in enumerate(zip(selected_terms, shuffled_types)):
            gen_func = self._type_to_gen.get(q_type)
            if gen_func:
                q = gen_func(i, term)
                if q:
                    questions.append(q)
                else:
                    # Этот случай уже маловероятен, так как мы добавили fallback в генераторы,
                    # но на всякий случай используем definition как запасной вариант
                    q = self._generate_definition(i, term)
                    questions.append(q)
            else:
                # Если нет функции для этого типа — используем definition
                q = self._generate_definition(i, term)
                questions.append(q)
        
        # Дополнительная гарантия: если по какой-то причине вопросов меньше, чем нужно
        # (например, terms меньше count), добавляем дополнительные вопросы
        # (но обычно terms = 57, так что это не требуется)
        while len(questions) < count and terms:
            # Берём случайный термин из оставшихся
            available_terms = [t for t in terms if t not in [q.context_source for q in questions]]
            if not available_terms:
                available_terms = terms  # если все термины использованы — повторяем
            
            term = random.choice(available_terms)
            q = self._generate_definition(len(questions), term)
            questions.append(q)
        
        return questions

    def _extract_def_sentence(self, term: str, text: str) -> str:
        """
        Извлекает предложение из 'text', содержащее термин 'term'.
        Использует морфологический поиск (pymorphy3) для нахождения термина
        в любой грамматической форме.
        """
        if not text:
            return ""
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Сначала пробуем точное совпадение (быстро)
        for s in sentences:
            if term.lower() in s.lower():
                return s.strip()
        
        # Если точное не найдено — пробуем морфологический поиск в каждом предложении
        for s in sentences:
            if _find_term_morph(term, s):
                return s.strip()
        
        # Если не нашли ни в одном предложении — возвращаем первое предложение
        return sentences[0].strip() if sentences else text[:200]

    def _get_term_ref(self, term: str) -> str:
        ref = self.retriever.get_term_context(term)
        if ref.startswith("Термин") and "не найден" in ref:
            # Fallback: TF-IDF поиск
            results = self.retriever.search(term, top_k=1)
            if results:
                ref = results[0]['text']
        return ref

    def _generate_definition(self, qid: int, term: str) -> Optional[QuestionData]:
        template = random.choice(self.DEFINITION_TEMPLATES)
        question_text = template.format(term=term.lower())
        reference = self._get_term_ref(term)
        return QuestionData(
            id=qid, type="definition",
            question_text=question_text,
            correct_answer=reference,
            context_source=f"Определение термина «{term}»",
        )

    def _generate_true_false(self, qid: int, term: str) -> Optional[QuestionData]:
        reference = self._get_term_ref(term)
        def_sent = self._extract_def_sentence(term, reference)

        if not def_sent:
            # УЛУЧШЕНИЕ: fallback на definition, если не удалось найти предложение
            return self._generate_definition(qid, term)

        # Истинное утверждение (из учебника) про термин `term`
        true_statement_about_term = def_sent
        if len(true_statement_about_term) > 250:
            true_statement_about_term = true_statement_about_term[:true_statement_about_term.rfind('.') + 1] if '.' in true_statement_about_term else true_statement_about_term[:250]

        # Решаем: показывать истинное или ложное утверждение
        # Используем морфологический поиск (pymorphy3), чтобы находить термины
        # в любой грамматической форме (Граф, Графом, Графа, и т.д.)
        term_found = _find_term_morph(term, def_sent)
        if random.random() < 0.5 and term_found:
            # Создаём ложное утверждение: заменяем термин на другой
            other_terms = [t for t in self.retriever.terms if t != term]
            wrong_term = random.choice(other_terms) if other_terms else "гиперграф"
            statement = _replace_term_morph(term, wrong_term, def_sent)
            is_true_statement = False
            
            # Для показа пользователю: объясняем, что утверждение ложно,
            # и показываем истинное утверждение про исходный термин
            display_text = (
                f"Утверждение ложно. Свойство «{true_statement_about_term}» "
                f"относится к термину «{term}», а не к «{wrong_term}»."
            )
        else:
            # Показываем истинное утверждение
            statement = def_sent
            is_true_statement = True
            
            # Для показа пользователю: показываем само истинное утверждение
            display_text = f"Утверждение верно: {true_statement_about_term}"

        # Обрезаем утверждение по предложению
        if len(statement) > 250:
            statement = statement[:statement.rfind('.') + 1] if '.' in statement else statement[:250]

        question_text = random.choice(self.TRUE_FALSE_TEMPLATES).format(statement=statement)

        # Улучшение:
        # - truth_value: "Верно" или "Неверно" (для валидации - сравниваем с ответом пользователя)
        # - correct_answer: развёрнутое объяснение для пользователя
        # - display_answer: развёрнутое объяснение для пользователя
        return QuestionData(
            id=qid, type="true_false",
            question_text=question_text,
            correct_answer=display_text,  # Развёрнутое объяснение
            truth_value="Верно" if is_true_statement else "Неверно",  # Для валидации
            display_answer=display_text,  # Для показа пользователю
            options=["Верно", "Неверно"],
            context_source=f"Определение термина «{term}»",
        )

    def _generate_multiple_choice(self, qid: int, term: str) -> Optional[QuestionData]:
        reference = self._get_term_ref(term)
        def_sent = self._extract_def_sentence(term, reference)

        if not def_sent:
            # УЛУЧШЕНИЕ: fallback на definition
            return self._generate_definition(qid, term)

        # УЛУЧШЕНИЕ: новый подход
        # Вопрос: "Какой термин соответствует определению: [definition]?"
        # Опции: термины (не определения)
        # Правильный ответ: сам термин

        # Обрезаем определение по длине
        if len(def_sent) > 200:
            def_sent = def_sent[:def_sent.rfind('.') + 1] if '.' in def_sent else def_sent[:200]

        # Дистракторы: другие термины
        other_terms = [t for t in self.retriever.terms if t != term]
        random.shuffle(other_terms)
        
        # Берём до 3 других терминов
        distractors = other_terms[:3]
        
        # Если недостаточно терминов — добавляем фоллбэки
        fallback_distractors = ["Гиперграф", "Мультиграф", "Ориентированный граф"]
        while len(distractors) < 3:
            fallback = fallback_distractors[len(distractors)]
            if fallback not in distractors and fallback != term:
                distractors.append(fallback)
            else:
                # Если фоллбэк уже используется или совпадает с термином, используем запасной
                distractors.append(f"Термин_{len(distractors)}")

        # Формируем опции
        options = [term] + distractors
        random.shuffle(options)

        # Текст вопроса: спрашиваем, какой термин соответствует определению
        question_text = f"Какой термин соответствует определению:\n«{def_sent}»?"

        return QuestionData(
            id=qid, type="multiple_choice",
            question_text=question_text,
            correct_answer=term,  # Теперь правильный ответ — сам термин
            display_answer=term,  # Для показа пользователю
            options=options,
            context_source=f"Определение термина «{term}»",
        )

    def _generate_fill_blank(self, qid: int, term: str) -> Optional[QuestionData]:
        """
        УЛУЧШЕННАЯ версия fill_blank с несколькими вариациями.
        
        Вариации:
        - V1 (full_term): заменить весь термин на __________
        - V2 (first_word): заменить первое слово термина (если составной)
        - V3 (last_word): заменить последнее слово термина (если составной)
        """
        reference = self._get_term_ref(term)
        def_sent = self._extract_def_sentence(term, reference)

        if not def_sent:
            return None

        # Если термин буквально не в предложении — fallback
        if term.lower() not in def_sent.lower():
            return self._generate_definition(qid, term)
        
        # Разбиваем термин на слова
        term_words = term.split()
        n_words = len(term_words)
        
        # Выбираем вариацию случайно
        if n_words >= 2:
            # Составной термин — доступны все вариации
            variation = random.choice(['full_term', 'first_word', 'last_word'])
        else:
            # Однословный — только полная замена
            variation = 'full_term'
        
        question_text = None
        correct_answer = None
        
        if variation == 'full_term':
            # V1: заменить весь термин
            question_text = re.sub(
                r'\b' + re.escape(term) + r'\b',
                '__________',
                def_sent,
                count=1,
                flags=re.IGNORECASE,
            )
            correct_answer = term
            
        elif variation == 'first_word' and n_words >= 2:
            # V2: заменить первое слово термина
            first_word = term_words[0]
            # Ищем вхождение первого слова в контексте термина
            # Пытаемся найти паттерн "первое_слово ... остальные_слова"
            pattern = re.escape(first_word) + r'(\s+\S+)*?\s+' + r'\s+'.join(re.escape(w) for w in term_words[1:])
            match = re.search(pattern, def_sent, flags=re.IGNORECASE)
            
            if match:
                # Заменяем только первое слово
                # Находим точную позицию первого слова в предложении
                full_match_text = match.group(0)
                # Заменяем первое слово на __________
                replacement = '__________' + full_match_text[len(first_word):]
                question_text = def_sent[:match.start()] + replacement + def_sent[match.end():]
                correct_answer = first_word
            else:
                # Fallback: заменить первое слово в термине
                # Простой подход: найти первое вхождение первого слова
                question_text = re.sub(
                    r'\b' + re.escape(first_word) + r'\b',
                    '__________',
                    def_sent,
                    count=1,
                    flags=re.IGNORECASE,
                )
                correct_answer = first_word
                
        elif variation == 'last_word' and n_words >= 2:
            # V3: заменить последнее слово термина
            last_word = term_words[-1]
            
            # Аналогичный подход: ищем весь термин, заменяем только последнее слово
            pattern = r'\s+'.join(re.escape(w) for w in term_words[:-1]) + r'\s+' + re.escape(last_word)
            match = re.search(pattern, def_sent, flags=re.IGNORECASE)
            
            if match:
                full_match_text = match.group(0)
                # Заменяем только последнее слово
                replacement = full_match_text[:-len(last_word)] + '__________'
                question_text = def_sent[:match.start()] + replacement + def_sent[match.end():]
                correct_answer = last_word
            else:
                # Fallback: простая замена последнего слова
                question_text = re.sub(
                    r'\b' + re.escape(last_word) + r'\b',
                    '__________',
                    def_sent,
                    count=1,
                    flags=re.IGNORECASE,
                )
                correct_answer = last_word
        
        # Safety: если что-то пошло не так — используем стандартную вариацию
        if question_text is None or correct_answer is None:
            question_text = re.sub(
                r'\b' + re.escape(term) + r'\b',
                '__________',
                def_sent,
                count=1,
                flags=re.IGNORECASE,
            )
            correct_answer = term

        return QuestionData(
            id=qid, type="fill_blank",
            question_text=question_text,
            correct_answer=correct_answer,
            context_source=f"Определение термина «{term}» (вариация: {variation})",
        )

    def _generate_matching(self, qid: int, term: str) -> Optional[QuestionData]:
        reference = self._get_term_ref(term)
        def_sent = self._extract_def_sentence(term, reference)

        if not def_sent:
            # УЛУЧШЕНИЕ: fallback на definition
            return self._generate_definition(qid, term)

        # Remove ALL occurrences of the term from the definition
        # Используем морфологический поиск (pymorphy3) вместо regex \b,
        # чтобы корректно обрабатывать падежи: Граф, Графом, Графа и т.д.
        masked = _mask_term_morph(term, def_sent, '_____')

        question_text = f"Какой термин соответствует следующему определению?\n\n«{masked}»"

        return QuestionData(
            id=qid, type="matching",
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
        """Проверяет ответ и возвращает bool (для обратной совместимости)."""
        result = TestValidator.check_answer_with_details(question, user_answer, retriever)
        return result.is_correct
    
    @staticmethod
    def check_answer_with_details(question: QuestionData, user_answer: str, 
                                   retriever=None) -> AnswerCheckResult:
        """
        Проверяет ответ и возвращает детальную информацию (включая score).
        
        Для открытых вопросов (definition): возвращает similarity_score, overlap_score, final_score.
        Для закрытых вопросов: score-поля None.
        """
        if question.type == "definition":
            return TestValidator._check_open_answer_with_details(
                question.correct_answer, user_answer, retriever
            )
        elif question.type == "true_false":
            # Для true_false используем truth_value (если есть), иначе correct_answer
            # Это позволяет:
            # - Валидировать ответ пользователя ("Верно"/"Неверно") с truth_value
            # - Показывать пользователю правильное утверждение из correct_answer
            correct_val = question.truth_value if question.truth_value is not None else question.correct_answer
            is_correct = user_answer.strip().lower() == correct_val.lower()
            return AnswerCheckResult(is_correct=is_correct)
        elif question.type in ["multiple_choice", "fill_blank", "matching"]:
            # Закрытые вопросы: точное сравнение
            is_correct = user_answer.strip().lower() == question.correct_answer.lower()
            return AnswerCheckResult(is_correct=is_correct)
        
        return AnswerCheckResult(is_correct=False)

    @staticmethod
    def _check_open_answer(reference: str, answer: str, retriever) -> bool:
        """TF-IDF проверка для открытых ответов (старая версия, для совместимости)."""
        result = TestValidator._check_open_answer_with_details(reference, answer, retriever)
        return result.is_correct
    
    @staticmethod
    def _check_open_answer_with_details(reference: str, answer: str, retriever) -> AnswerCheckResult:
        """TF-IDF проверка для открытых ответов с детальной информацией о score."""
        if not answer.strip():
            return AnswerCheckResult(
                is_correct=False,
                similarity_score=0.0,
                overlap_score=0.0,
                final_score=0.0,
            )
        
        similarity_score = 0.0
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            vecs = retriever.vectorizer.transform([answer, reference])
            similarity_score = float(cosine_similarity(vecs[0], vecs[1])[0][0])
        except Exception:
            similarity_score = 0.0

        # Проверка ключевых слов
        ref_words = set(reference.lower().split())
        ans_words = set(answer.lower().split())
        overlap_score = len(ref_words & ans_words) / max(len(ref_words), 1)

        final_score = 0.6 * similarity_score + 0.4 * overlap_score
        is_correct = final_score >= 0.12  # мягкий порог

        return AnswerCheckResult(
            is_correct=is_correct,
            similarity_score=similarity_score,
            overlap_score=overlap_score,
            final_score=final_score,
            passing_threshold=0.12,
        )

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
