"""
Retriever v9 — TF-IDF + pymorphy3 + бустинг заголовков + структурированные чанки.

Оптимизация: леммы всех слов вычисляются ОДИН РАЗ при старте,
затем используются для всех 57 терминов без повторных вызовов morph.parse.
Время старта: ~10 секунд.

Поддерживает два паттерна определений:
  1. «Маршрутом называется [...]» — термин перед маркером, субъектный падеж
  2. «[...] называется циклом»   — термин сразу после маркера (≤15 символов)

Новое в v9:
  - Структурированные чанки (Chunk) с заголовками раздела
  - Бустинг по заголовкам (×1.5) и терминам (×1.3)
  - Fallback-поиск с расширением запроса при низкой уверенности
  - get_context_for_llm() для подготовки контекста LLM
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────── Датаклассы ─────────────────────────────────────────

@dataclass
class Chunk:
    """Структурированный фрагмент текста с заголовком раздела."""
    index: int
    title: str               # заголовок раздела
    level: int               # уровень заголовка (1, 2, 3)
    text: str                # содержимое чанка
    terms: List[str] = field(default_factory=list)  # найденные термины


@dataclass
class SearchResult:
    """Результат поиска с мета-информацией."""
    chunk: Chunk
    score: float
    boosted: bool = False
    term_match: Optional[str] = None


# ──────────────────────── Глобальные константы ───────────────────────────────

morph = pymorphy3.MorphAnalyzer()

STOPWORDS = {
    'и', 'в', 'во', 'не', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она',
    'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'из', 'от', 'это',
    'если', 'при', 'или', 'об', 'для', 'до', 'её', 'им', 'без', 'под', 'через', 'над',
    'тогда', 'когда', 'также', 'потому', 'который', 'которая', 'которое', 'которые',
    'такой', 'такое', 'такая', 'такие', 'этот', 'эта', 'эти', 'тот', 'та', 'те',
    'нет', 'есть', 'быть', 'дать', 'мочь', 'пусть', 'ещё',
    'уже', 'только', 'свой', 'их', 'мы', 'нас', 'вас', 'ним', 'них',
    'что', 'такое', 'чем', 'какой', 'какая', 'зачем', 'почему', 'где',
    'дайте', 'объясните', 'опишите', 'расскажите', 'определение', 'понятие',
}

DEF_MARKERS = ['называется', 'называют', 'определяется', '— это', 'Def =']
NON_SUBJECT_CASES = {'gent', 'datv', 'loct', 'accs'}

# Кэш морфологического разбора слов (global)
_morph_cache: Dict[str, Tuple[str, Optional[str]]] = {}  # word → (lemma, case)


# ──────────────────────── Хелперы ────────────────────────────────────────────

def _parse_word(word: str) -> Tuple[str, Optional[str]]:
    """Разбор слова с кэшированием: возвращает (lemma, case)."""
    w = word.lower()
    if w not in _morph_cache:
        p = morph.parse(w)
        if p:
            _morph_cache[w] = (p[0].normal_form, p[0].tag.case)
        else:
            _morph_cache[w] = (w, None)
    return _morph_cache[w]


def lemmatize(text: str) -> str:
    tokens = re.findall(r'[а-яёА-ЯЁ]+', text.lower())
    lemmas = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        lemma, _ = _parse_word(tok)
        if len(lemma) > 2:
            lemmas.append(lemma)
        # Если оригинал слова отличается от леммы и не стоп-слово — добавляем тоже
        if tok != lemma and len(tok) > 2 and tok not in STOPWORDS:
            lemmas.append(tok)
    return ' '.join(lemmas)


def clean_text(text: str) -> str:
    """Очищает текст от LaTeX-разметки для читаемого отображения."""
    if not text:
        return text
    # 1. Удаляем backslash-escape перед пунктуацией (включая |)
    text = re.sub(r'\\([.*\[\]()!?\-−:=|])', r'\1', text)
    # 2. \... → ...
    text = text.replace('\\...', '...')
    # 3. Двойной backslash \\ (LaTeX set-difference) → убираем
    text = text.replace('\\\\', '')
    # 4. Оставшиеся \ перед пробелом/концом строки
    text = re.sub(r'\\(\s)', r'\1', text)
    text = re.sub(r'\\$', '', text)
    # 5. Заменяем Def = на «— это»
    text = re.sub(r'Def\s*(=)', '— это', text)
    # 6. Угловые скобки ⟨⟩ → < > (совместимость)
    text = text.replace('⟨', '<').replace('⟩', '>')
    # 7. /∈ → ∉ (LaTeX \notin в кривой конвертации)
    text = text.replace('/∈', '∉')
    # 8. Комбинирующий слэш ̸ + = → ≠, ̸ + < → ≮, ̸ + > → ≯
    text = re.sub(r'\u0338=', '≠', text)
    text = re.sub(r'\u0338<', '≮', text)
    text = re.sub(r'\u0338>', '≯', text)
    text = re.sub(r'\u0338', '', text)  # оставшиеся — удаляем
    # 8. Чистим повторы пробелов
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _split_into_chunks_flat(text: str, min_len: int = 80, max_len: int = 500) -> List[str]:
    """
    Плоское разбиение текста на чанки (обратная совместимость).
    Удаляет заголовки markdown, дробит по параграфам и предложениям.
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'#+\s', '', text)
    text = re.sub(r'\*+', '', text)

    paragraphs = text.split('\n\n')
    chunks = []

    for para in paragraphs:
        para = para.strip()
        para = re.sub(r'\s+', ' ', para)
        if len(para) < min_len:
            continue

        if len(para) <= max_len:
            chunks.append(para)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= max_len:
                    current = (current + " " + sent).strip()
                else:
                    if len(current) >= min_len:
                        chunks.append(current)
                    current = sent
            if len(current) >= min_len:
                chunks.append(current)

    return chunks


def segment_chapter(text: str, terms: Optional[List[str]] = None) -> List[Chunk]:
    """
    Разбивает текст главы на структурированные чанки с заголовками разделов.

    - Определяет заголовки ## и ###
    - Чанки получают title (текст заголовка) и level (уровень вложенности)
    - Раздел >500 символов дробится по абзацам
    - Первые строки без заголовка относятся к "Введению" (level=1)
    - В каждом чанке заполняется terms — найденные вхождения из terms
    """
    if terms is None:
        terms = []

    def _find_terms(txt: str) -> List[str]:
        txt_lower = txt.lower()
        return [t for t in terms if t.lower() in txt_lower]

    # Нормализация: множественные переносы → двойной
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Сначала разбиваем на абзацы (по двойному переносу)
    paragraphs = text.split('\n\n')
    # Склеиваем строки внутри каждого абзаца
    merged_paragraphs = []
    for p in paragraphs:
        lines = [line.strip() for line in p.split('\n') if line.strip()]
        if lines:
            merged_paragraphs.append(' '.join(lines))

    chunks: List[Chunk] = []
    current_title = "Введение"
    current_level = 1
    current_content: List[str] = []

    def _flush_section() -> None:
        """Преобразовать накопленное содержание в чанки."""
        nonlocal current_content
        if not current_content:
            return

        section_text = ' '.join(current_content)
        section_text = re.sub(r'\*+', '', section_text)
        section_text = re.sub(r'\s+', ' ', section_text).strip()
        if not section_text or len(section_text) < 80:
            current_content = []
            return

        # Если весь раздел влезает в 500 символов — один чанк
        if len(section_text) <= 500:
            detected_terms = _find_terms(section_text)
            chunks.append(Chunk(
                index=len(chunks),
                title=clean_text(current_title),
                level=current_level,
                text=clean_text(section_text),
                terms=detected_terms,
            ))
        else:
            parts = []
            current_part = ""
            for paragraph in current_content:
                paragraph = re.sub(r'\s+', ' ', paragraph).strip()
                if not paragraph or len(paragraph) < 40:
                    continue
                if len(current_part) + len(paragraph) <= 500:
                    current_part = (current_part + ' ' + paragraph).strip()
                else:
                    if current_part and len(current_part) >= 80:
                        parts.append(current_part)
                    current_part = paragraph
            if current_part and len(current_part) >= 80:
                parts.append(current_part)

            for part in parts:
                detected_terms = _find_terms(part)
                chunks.append(Chunk(
                    index=len(chunks),
                    title=clean_text(current_title),
                    level=current_level,
                    text=clean_text(part),
                    terms=detected_terms,
                ))

        current_content = []

    for para in merged_paragraphs:
        # Определяем заголовки ## и ###
        header_match = re.match(r'^(#{2,3})\s+(.+)$', para)
        if header_match:
            _flush_section()
            level = len(header_match.group(1))
            current_title = header_match.group(2).strip()
            current_level = level
        else:
            current_content.append(para)

    _flush_section()

    # Индексы уже правильные (присваиваются при создании)
    return chunks


# ─── Предварительно вычисленные данные о предложениях ────────────────────────
# Структура: (sentence_text, lemma_set, word_tokens_with_pos)
# word_tokens_with_pos: list of (start, end, lemma, case)
SentenceData = Tuple[str, frozenset, List[Tuple[int, int, str, Optional[str]]]]


def _precompute_sentences(chunks: List[str]) -> List[List[SentenceData]]:
    """
    Для каждого чанка разбиваем на предложения и вычисляем леммы ОДИН РАЗ.
    Используется при построении term_index.
    """
    result = []
    for chunk in chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        chunk_sents = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 25:
                continue
            s_lower = sent.lower()
            # Токены с позициями
            tokens = []
            for m in re.finditer(r'[а-яёА-ЯЁ]+', s_lower):
                lemma, case = _parse_word(m.group())
                tokens.append((m.start(), m.end(), lemma, case))
            lemma_set = frozenset(t[2] for t in tokens)
            chunk_sents.append((sent, lemma_set, tokens))
        result.append(chunk_sents)
    return result


def _find_term_words(term_lemmas, tokens: List) -> List[List[Tuple]]:
    """
    Для составных терминов: находит все вхождения КАЖДОГО слова термина отдельно.
    
    Возвращает: список списков токенов для каждого слова термина.
    Например: для термина из 2 слов вернёт [[токены_слова1], [токены_слова2]]
    """
    is_nested = term_lemmas and isinstance(term_lemmas[0], list)
    n_words = len(term_lemmas)
    
    result = []
    for w in range(n_words):
        alts = term_lemmas[w] if is_nested else [term_lemmas[w]]
        word_tokens = []
        for t in tokens:
            if t[2] in alts:
                word_tokens.append(t)
        result.append(word_tokens)
    return result


def _generate_term_combinations(word_token_lists: List[List]) -> List[Tuple[int, int, Optional[str], float]]:
    """
    Генерирует все возможные комбинации токенов для составного термина.
    
    word_token_lists: [[токены_слова1], [токены_слова2], ...]
    
    Возвращает: список (start_pos, end_pos, case, compactness_score)
    где compactness_score — насколько близко слова стоят друг к другу (1.0 = идеально, 0.0 = очень далеко)
    """
    from itertools import product
    
    # Проверяем, есть ли вхождения для всех слов
    for w, w_tokens in enumerate(word_token_lists):
        if not w_tokens:
            return []
    
    results = []
    
    # Перебираем все комбинации: одна позиция для первого слова, одна для второго и т.д.
    for combo in product(*word_token_lists):
        # combo = (token_word1, token_word2, ...)
        
        # Проверяем, что слова идут в правильном порядке (токены не пересекаются)
        # Разрешаем ЛЮБОЙ порядок для гибкости, но оцениваем компактность
        positions = [(t[0], t[1]) for t in combo]
        min_pos = min(p[0] for p in positions)
        max_pos = max(p[1] for p in positions)
        
        # Оцениваем компактность: сколько символов занимает «вхождение» относительно суммы длин слов
        span_len = max_pos - min_pos
        
        if span_len == 0:
            compactness = 1.0
        else:
            # Чем меньше «лишнего» пространства между словами, тем лучше
            # Максимальный разрешённый span: 100 символов (для вставок вроде «(дизъюнктное)»)
            if span_len > 150:
                compactness = 0.0
            else:
                compactness = max(0.0, 1.0 - span_len / 150.0)
        
        if compactness > 0.0:
            # Берём падеж ПОСЛЕДНЕГО слова в термине (как наиболее значимого)
            # или того слова, которое ближе всего к концу «вхождения»
            last_word_token = combo[-1]
            results.append((min_pos, max_pos, last_word_token[3], compactness))
    
    # Сортируем по компактности (лучшие первые)
    results.sort(key=lambda x: x[3], reverse=True)
    return results


def _find_all_term_occurrences(term_lemmas, tokens: List) -> List[Tuple[int, int, Optional[str]]]:
    """
    Находит все ВОЗМОЖНЫЕ позиции термина в токенах.
    
    Для однословных терминов: все прямые вхождения.
    Для составных терминов: все комбинации слов с разумными промежутками (до 150 символов).
    Это позволяет находить:
      - «Объединение (дизъюнктное) графов» (вставка в скобках)
      - «Стягивание правильного подграфа» (вставка прилагательного)
      - «вершина ... изолированной» (обратный порядок + промежуток с маркером)
    
    Возвращает: список (start_pos, end_pos, case) для каждого возможного вхождения.
    """
    is_nested = term_lemmas and isinstance(term_lemmas[0], list)
    n_words = len(term_lemmas)
    
    if n_words == 1:
        # Однословный термин — ищем все вхождения
        alts = term_lemmas[0] if is_nested else [term_lemmas[0]]
        results = []
        for t in tokens:
            if t[2] in alts:
                results.append((t[0], t[1], t[3]))
        return results
    else:
        # Составной термин — ИЩЕМ ВСЕ КОМБИНАЦИИ слов
        word_token_lists = _find_term_words(term_lemmas, tokens)
        combinations = _generate_term_combinations(word_token_lists)
        
        # Возвращаем без compactness_score (он нужен был только для сортировки)
        return [(c[0], c[1], c[2]) for c in combinations]


def _score_sentence_v2(term_lemmas, sent_data: SentenceData) -> float:
    """
    УЛУЧШЕННАЯ версия: оценивает предложение как определение термина.
    
    Особенности:
    1. Для однословных терминов — СТРОГИЕ проверки (чтобы не ловить случайные упоминания)
    2. Для составных терминов — гибкий поиск: разрешён любой порядок слов, большие промежутки
    3. Поддержка «перевёрнутых» определений: «вершина называется изолированной»
       при термине «Изолированная вершина» (слова термина до и после маркера)
    4. Маркеры: 'называется', 'называются', '— это', 'даёт граф' и др.
    """
    sent, lemma_set, tokens = sent_data
    s_lower = sent.lower()
    
    # Определяем тип термина
    is_nested = term_lemmas and isinstance(term_lemmas[0], list)
    n_words = len(term_lemmas)
    is_single_word = (n_words == 1)

    # Проверяем, что все слова термина есть в предложении
    if is_nested:
        for word_alts in term_lemmas:
            if not any(alt in lemma_set for alt in word_alts):
                return 0.0
    else:
        if not all(tl in lemma_set for tl in term_lemmas):
            return 0.0
    
    # ============== МАРКЕРЫ ОПРЕДЕЛЕНИЙ ==============
    
    standard_markers = [
        ('называется', 0.6, 0.75),
        ('называются', 0.6, 0.75),  # мн.ч.
        ('называют', 0.6, 0.75),
        ('определяется', 0.6, 0.75),
        ('— это', 0.7, 0.8),      # тире-определитель
        ('--- это', 0.7, 0.8),
    ]
    
    operation_markers = [
        ('даёт граф', 0.55),
        ('дает граф', 0.55),
    ]
    
    best_score = 0.0
    
    # ============== ЕСЛИ ОДНОСЛОВНЫЙ ТЕРМИН — СТРОГИЕ ПРАВИЛА ==============
    
    if is_single_word:
        # Для однословных терминов ищем все вхождения
        alts = term_lemmas[0] if is_nested else [term_lemmas[0]]
        term_occurrences = []
        for t in tokens:
            if t[2] in alts:
                # Токен имеет формат (start, end, lemma, case) — берём только нужные
                term_occurrences.append((t[0], t[1], t[3]))
        
        # Для однословных: очень высокие требования к близости к маркеру
        # Это чтобы «Вершина» и «Ребро» не находились в определении смежности
        
        for marker, pattern1_base, pattern2_base in standard_markers:
            marker_pos = s_lower.find(marker)
            if marker_pos < 0:
                continue
            marker_end = marker_pos + len(marker)
            
            for (term_start, term_end, word_case) in term_occurrences:
                # ПАТТЕРН 1: термин ПЕРЕД маркером (строгий gap)
                if term_end <= marker_pos:
                    gap = marker_pos - term_end
                    if gap > 50:  # ОЧЕНЬ СТРОГО для однословных!
                        continue
                    if word_case in NON_SUBJECT_CASES:
                        continue
                    start_bonus = 0.3 if term_start <= 15 else 0.0
                    gap_factor = max(0.4, 1.0 - gap / 50.0)
                    score = min(1.0, gap_factor + start_bonus)
                    if score > best_score:
                        best_score = score
                
                # ПАТТЕРН 2: термин ПОСЛЕ маркером (также строго)
                if marker_end < term_start:
                    gap = term_start - marker_end
                    if gap <= 40:  # Строже, чем для составных
                        score = pattern2_base + max(0.0, 0.15 - gap / 300.0)
                        if score > best_score:
                            best_score = score
        
        # Маркеры операций «даёт граф» для однословных обычно не актуальны
        # (операции — составные термины)
        
        return best_score
    
    # ============== ЕСЛИ СОСТАВНОЙ ТЕРМИН — ГИБКИЕ ПРАВИЛА ==============
    
    # Для составных терминов находим все слова отдельно (для разорванных вхождений)
    word_token_lists = _find_term_words(term_lemmas, tokens)
    if not word_token_lists or any(len(wl) == 0 for wl in word_token_lists):
        return 0.0
    
    # Также получаем «компактные» вхождения через старую функцию
    compact_occurrences = _find_all_term_occurrences(term_lemmas, tokens)
    
    # ============== ПРОВЕРКА СТАНДАРТНЫХ МАРКЕРОВ ==============
    
    for marker, pattern1_base, pattern2_base in standard_markers:
        marker_pos = s_lower.find(marker)
        if marker_pos < 0:
            continue
        marker_end = marker_pos + len(marker)
        
        # --- ПАТТЕРН A: КОМПАКТНОЕ ВХОЖДЕНИЕ (все слова рядом) ---
        for (term_start, term_end, word_case) in compact_occurrences:
            # Подпаттерн A1: всё вхождение ПЕРЕД маркером
            if term_end <= marker_pos:
                gap = marker_pos - term_end
                if gap > 120:
                    continue
                # Для составных падежная проверка менее строгая (смотрим на последнее слово)
                start_bonus = 0.3 if term_start <= 15 else 0.1 if term_start <= 40 else 0.0
                gap_factor = max(0.4, 1.0 - gap / 120.0)
                score = min(1.0, gap_factor + start_bonus)
                if score > best_score:
                    best_score = score
            
            # Подпаттерн A2: всё вхождение ПОСЛЕ маркера
            if marker_end < term_start:
                gap = term_start - marker_end
                if gap <= 80:
                    score = pattern2_base + max(0.0, 0.15 - gap / 600.0)
                    if score > best_score:
                        best_score = score
        
        # --- ПАТТЕРН B: РАЗОРВАННОЕ ВХОЖДЕНИЕ (перевёрнутое определение) ---
        # Пример: термин «Изолированная вершина», текст: «вершина называется изолированной»
        # «вершина» (слово 2 термина) — ДО маркера
        # «изолированной» (слово 1 термина) — ПОСЛЕ маркера
        
        # Идея: проверить, что ЕСТЬ вхождения каждого слова и до, и после маркера
        words_before = []
        words_after = []
        
        for w, w_tokens in enumerate(word_token_lists):
            has_before = any(t[1] <= marker_pos for t in w_tokens)
            has_after = any(t[0] > marker_end for t in w_tokens)
            if has_before:
                words_before.append(w)
            if has_after:
                words_after.append(w)
        
        # Если есть слова и до, и после маркера — это потенциально «перевёрнутое» определение
        if len(words_before) > 0 and len(words_after) > 0:
            # Дополнительно: проверим, что распределение логичное
            # Например, для «Изолированная вершина» («прил. + сущ.»):
            #   В тексте: «сущ. называется прил._твор.»
            #   Существительное (второе слово термина) — ДО маркера
            #   Прилагательное (первое слово термина) — ПОСЛЕ маркера
            
            # Найдём ближайшее вхождение слова ДО маркера и ближайшее ПОСЛЕ
            min_dist_before = float('inf')
            min_dist_after = float('inf')
            
            for w_tokens in word_token_lists:
                for t in w_tokens:
                    if t[1] <= marker_pos:
                        dist = marker_pos - t[1]
                        if dist < min_dist_before:
                            min_dist_before = dist
                    if t[0] > marker_end:
                        dist = t[0] - marker_end
                        if dist < min_dist_after:
                            min_dist_after = dist
            
            # Если оба расстояния разумные — высокий score
            if min_dist_before < 80 and min_dist_after < 80:
                # Чем меньше суммарное расстояние — тем лучше
                total_gap = min_dist_before + min_dist_after
                gap_factor = max(0.5, 1.0 - total_gap / 160.0)
                # Бонус за «перевёрнутое» определение (часто это именно определение!)
                inversion_bonus = 0.15
                score = min(1.0, 0.7 + gap_factor * 0.2 + inversion_bonus)
                if score > best_score:
                    best_score = score
    
    # ============== ПРОВЕРКА МАРКЕРОВ ОПЕРАЦИЙ «даёт граф» ==============
    
    for marker, base_score in operation_markers:
        marker_pos = s_lower.find(marker)
        if marker_pos < 0:
            continue
        
        # Для операций термин должен быть ПЕРЕД «даёт граф»
        # Используем компактные вхождения
        for (term_start, term_end, _) in compact_occurrences:
            if term_end <= marker_pos:
                gap = marker_pos - term_end
                if gap < 250:  # Операции могут иметь длинные условия
                    score = base_score + max(0.0, 0.15 - gap / 1800.0)
                    if score > best_score:
                        best_score = score
    
    return best_score


def _score_sentence(term_lemmas, sent_data: SentenceData) -> float:
    """Обёртка: вызывает улучшенную v2 версию."""
    return _score_sentence_v2(term_lemmas, sent_data)


# ──────────────────────── Retriever ──────────────────────────────────────────

class Retriever:
    def __init__(self, chapter_path: str, terms_path: str):
        with open(chapter_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        with open(terms_path, 'r', encoding='utf-8') as f:
            self.terms = [line.strip() for line in f if line.strip()]

        # Структурированные чанки
        self._chunks = segment_chapter(raw_text, self.terms)
        print(f'[Retriever] {len(self._chunks)} чанков')

        print('[Retriever] Лемматизация чанков + предложений...')
        self.chunks_lemmatized = [lemmatize(c.text) for c in self._chunks]
        # Предвычисляем предложения (кэш morph заполняется при lemmatize выше)
        self._chunk_sentences = _precompute_sentences([c.text for c in self._chunks])

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=15000,
        )
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks_lemmatized)
        print(f'[Retriever] TF-IDF матрица: {self.chunk_vectors.shape}')

        self._terms_lemmatized = {t: lemmatize(t) for t in self.terms}
        self._build_term_index()

    @property
    def chunks(self) -> List[str]:
        """Свойство для обратной совместимости: возвращает список текстов чанков."""
        return [c.text for c in self._chunks]

    # ── Ручные определения для проблемных терминов ────────────────────────────

    # ── Ручные определения для терминов, которые НЕЛЬЗЯ извлечь автоматически ──
    # 
    # Причина: в тексте учебника нет явного определения с маркером для этих понятий:
    # - Вершина, Ребро — определяются только неявно через множества V и E в определении Графа
    # - Перестановка — используется в тексте, но явно не определяется
    #
    # Все остальные 18 из бывших MANUAL_DEFS теперь извлекаются автоматически.
    
    MANUAL_DEFS = {
        'Вершина': (
            'Вершина (или узел) — базовый элемент графа; '
            'непустое множество V называется множеством вершин графа G(V, E).'
        ),
        'Ребро': (
            'Ребро — элемент множества E двухэлементных подмножеств множества V; '
            'соединяет две вершины графа.'
        ),
        'Перестановка': (
            'Перестановка — взаимно однозначное отображение '
            'конечного множества на себя.'
        ),
    }

    # ── Построение индекса определений ──────────────────────────────────────

    def _build_term_index(self):
        """
        Строим индекс определений за O(N_chunks × N_sentences_per_chunk),
        переиспользуя предвычисленные леммы предложений.
        """
        self.term_index: Dict[str, Dict] = {}

        # Предвычисляем леммы терминов (со всеми вариантами от pymorphy3)
        term_lemmas_map = {}
        for term in self.terms:
            words = term.lower().split()
            word_alts = []  # список списков: для каждого слова — все варианты лемм
            for w in words:
                alts = set()
                lemma, _ = _parse_word(w)
                alts.add(lemma)
                for p in morph.parse(w):
                    alt = p.normal_form
                    if len(alt) > 2:
                        alts.add(alt)
                word_alts.append(list(alts))
            term_lemmas_map[term] = word_alts

        for term in self.terms:
            term_lemmas = term_lemmas_map[term]
            found = False

            for i, sent_list in enumerate(self._chunk_sentences):
                for sent_data in sent_list:
                    score = _score_sentence(term_lemmas, sent_data)
                    if score >= 0.5:
                        sent_text = sent_data[0]
                        # Ищем название параграфа — первые 1-2 предложения чанка
                        chunk = self._chunks[i]
                        prefix = chunk.text[:chunk.text.index(sent_text)] if sent_text in chunk.text else ''
                        if prefix.strip():
                            title_part = prefix.strip()[:150]
                            self.term_index[term] = {
                                'text': sent_text,
                                'title': title_part,
                                'score': 0.42,
                                'chunk_id': i,
                                'def_score': score,
                            }
                        else:
                            self.term_index[term] = {
                                'text': sent_text,
                                'score': 0.42,
                                'chunk_id': i,
                                'def_score': score,
                            }
                        found = True
                        break
                if found:
                    break

            if not found:
                results = self._tfidf_search(term, top_k=1)
                if results:
                    txt = results[0].get('text', '')
                    # Берём первое предложение результата как определение
                    first_sent = txt.split('.')[0] + '.' if '.' in txt else txt
                    self.term_index[term] = {
                        'text': first_sent,
                        'score': results[0].get('score', 0.3),
                        'chunk_id': results[0].get('chunk_id', 0),
                        'def_score': 0.0,
                    }

        # Ручные переопределения для терминов, которые нельзя извлечь автоматически
        for term, def_text in self.MANUAL_DEFS.items():
            self.term_index[term] = {
                'text': def_text,
                'score': 0.5,
                'chunk_id': 0,
                'def_score': 1.0,
            }

        indexed = sum(1 for v in self.term_index.values() if v.get('def_score', 0) >= 0.5)
        print(f'[Retriever] Индекс: {len(self.term_index)} терминов ({indexed} точных)')

    # ── TF-IDF ──────────────────────────────────────────────────────────────

    def _tfidf_search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_lem = lemmatize(query)
        if not query_lem.strip():
            query_lem = query.lower()

        query_vec = self.vectorizer.transform([query_lem])
        scores = cosine_similarity(query_vec, self.chunk_vectors)[0].copy()

        for i, chunk in enumerate(self.chunks):
            cl = chunk.lower()
            if any(m.lower() in cl for m in DEF_MARKERS):
                scores[i] *= 1.25

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0.03:
                results.append({
                    'text': self.chunks[idx],
                    'score': float(scores[idx]),
                    'chunk_id': int(idx),
                })
        return results

    # ── Определение термина в запросе ──────────────────────────────────────

    def _detect_term_in_query(self, query: str) -> Optional[str]:
        """Длинные термины с более высоким приоритетом (Связный граф > Граф)."""
        query_lower = query.lower()
        query_lem = lemmatize(query)

        for term in sorted(self.terms, key=lambda t: len(t), reverse=True):
            if term.lower() in query_lower:
                return term
            term_lem = self._terms_lemmatized.get(term, '')
            if term_lem and len(term_lem) > 3 and term_lem in query_lem:
                return term

        return None

    # ── Поиск (основной) ───────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Поиск с бустингом заголовков и терминов.
        Возвращает List[Dict] для обратной совместимости.
        Внутри использует search_ranked().
        """
        ranked = self.search_ranked(query, top_k=top_k)
        return [
            {
                'text': r.chunk.text,
                'score': r.score,
                'chunk_id': r.chunk.index,
            }
            for r in ranked
        ]

    def search_ranked(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Поиск с возвратом SearchResult.

        1. Определяем термин в запросе → ищем в term_index
        2. TF-IDF поиск (через _tfidf_search)
        3. Бустинг заголовков: query содержит название заголовка чанка → ×1.5
        4. Бустинг терминов: query совпадает с термином из terms.txt → ×1.3
        5. Если найден термин в term_index — форсируем его на top с высоким score
        6. Сортировка по убыванию score, возврат топ-k
        """
        query_lower = query.lower()
        detected_term = self._detect_term_in_query(query)

        # 1. TF-IDF (берём с запасом)
        tfidf_results = self._tfidf_search(query, top_k=top_k * 2)

        results: List[SearchResult] = []
        seen_chunk_ids: set = set()

        # 2. Если обнаружен термин из term_index — добавляем его принудительно
        if detected_term and detected_term in self.term_index:
            term_data = self.term_index[detected_term]
            chunk_id = term_data.get('chunk_id')
            if chunk_id is not None and 0 <= chunk_id < len(self._chunks):
                chunk = self._chunks[chunk_id]
                # Высокий score для точного совпадения термина
                term_score = max(0.6, term_data.get('score', 0.5))
                results.append(SearchResult(
                    chunk=chunk,
                    score=term_score,
                    term_match=detected_term,
                ))
                seen_chunk_ids.add(chunk_id)

        # 3. Обрабатываем TF-IDF результаты
        for r in tfidf_results:
            chunk_id = r['chunk_id']
            if chunk_id in seen_chunk_ids:
                continue
            chunk = self._chunks[chunk_id]
            score = r['score']
            term_match: Optional[str] = None

            # Бустинг заголовка
            if chunk.title.lower() in query_lower:
                score *= 1.5

            # Бустинг термина
            if detected_term:
                score *= 1.3
                term_match = detected_term

            results.append(SearchResult(
                chunk=chunk,
                score=score,
                term_match=term_match,
            ))
            seen_chunk_ids.add(chunk_id)

        # 4. Сортировка и топ-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def search_with_fallback(self, query: str, min_score: float = 0.15) -> List[SearchResult]:
        """
        Поиск с fallback при низкой уверенности.

        1. Выполнить search_ranked(query, top_k=3)
        2. Если лучший результат < min_score:
           - Расширить запрос лемматизированными формами соседних терминов
           - Повторить поиск
           - Если всё ещё < min_score: пометить boosted=True, score = max(score, 0.1)
        3. Вернуть результаты
        """
        results = self.search_ranked(query, top_k=3)

        if results and results[0].score < min_score:
            # Расширяем запрос: ищем термины, леммы которых пересекаются с леммами запроса
            query_lemmas = set(lemmatize(query).split())
            related_terms = []
            for term in self.terms:
                term_lemmas = set(lemmatize(term).split())
                if query_lemmas & term_lemmas:  # есть пересечение
                    related_terms.append(term)

            if related_terms:
                expanded_query = query + " " + " ".join(related_terms)
                results = self.search_ranked(expanded_query, top_k=3)

            # Если всё ещё низкий score — помечаем boosted
            if results and results[0].score < min_score:
                for r in results:
                    r.boosted = True
                    r.score = max(r.score, 0.1)

        return results

    # ── Контекст термина ───────────────────────────────────────────────────

    def get_term_context(self, term: str) -> str:
        """Возвращает текстовый контекст термина (строка)."""
        if term in self.term_index:
            return self.term_index[term]['text']
        results = self._tfidf_search(term, top_k=1)
        return results[0]['text'] if results else f"Термин '{term}' не найден."

    def get_context_for_llm(self, term: str) -> Dict:
        """
        Подготовить контекст для LLM-запроса.

        Возвращает:
            term            — искомый термин
            definition      — определение/текст, где найден термин
            section_title   — заголовок раздела, в котором найден термин
            surrounding_text — полный текст чанка с термином
        """
        term_info = self.term_index.get(term, {})

        if term_info:
            chunk_id = term_info.get('chunk_id')
            if chunk_id is not None and 0 <= chunk_id < len(self._chunks):
                chunk = self._chunks[chunk_id]
                definition = term_info.get('text', chunk.text)
                section_title = chunk.title
                surrounding_text = chunk.text
            else:
                definition = term_info.get('text', '')
                section_title = "Основные понятия"
                surrounding_text = definition
        else:
            results = self._tfidf_search(term, top_k=1)
            if results:
                chunk_id = results[0]['chunk_id']
                chunk = self._chunks[chunk_id]
                definition = results[0]['text']
                section_title = chunk.title
                surrounding_text = chunk.text
            else:
                definition = ""
                section_title = ""
                surrounding_text = ""

        return {
            "term": term,
            "definition": definition,
            "section_title": section_title,
            "surrounding_text": surrounding_text,
        }
