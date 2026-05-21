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
    return ' '.join(lemmas)


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
        lines = [l.strip() for l in p.split('\n') if l.strip()]
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
                title=current_title,
                level=current_level,
                text=section_text,
                terms=detected_terms,
            ))
        else:
            # Дробим: каждый абзац — отдельный чанк (или по 500 символов)
            # Здесь абзацы уже соединены, дробим по предложениям
            import re as _re
            # Пробуем разбить по границам абзацев (двойные переносы уже убраны,
            # поэтому дробим по ~500 символов, не разрывая предложения)
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
                    title=current_title,
                    level=current_level,
                    text=part,
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


def _score_sentence(term_lemmas: List[str], sent_data: SentenceData) -> float:
    """
    Оценивает предложение как определение термина, используя предвычисленные данные.
    term_lemmas: леммы слов термина.
    sent_data: (sentence, lemma_set, tokens).
    """
    sent, lemma_set, tokens = sent_data

    # Все леммы термина должны присутствовать
    if not all(tl in lemma_set for tl in term_lemmas):
        return 0.0

    # Находим первый токен с леммой = term_lemmas[0]
    first_lemma = term_lemmas[0]
    first_token = None
    for t in tokens:
        if t[2] == first_lemma:
            first_token = t
            break

    if first_token is None:
        return 0.0

    term_start, term_end, _, word_case = first_token
    s_lower = sent.lower()

    for marker in ['называется', 'называют', 'определяется']:
        marker_pos = s_lower.find(marker)
        if marker_pos < 0:
            continue
        marker_end = marker_pos + len(marker)

        # Паттерн 1: термин ПЕРЕД маркером
        if term_end <= marker_pos:
            gap = marker_pos - term_end
            if gap > 80:
                continue
            # Генитив/датив/локатив/аккузатив → не субъект → пропускаем
            if word_case in NON_SUBJECT_CASES:
                continue
            start_bonus = 0.3 if term_start <= 10 else 0.0
            return min(1.0, max(0.4, 1.0 - gap / 100.0) + start_bonus)

        # Паттерн 2: термин ПОСЛЕ маркера (≤15 символов)
        if marker_end < term_start and term_start - marker_end <= 15:
            return 0.75

    return 0.0


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

    # ── Построение индекса определений ──────────────────────────────────────

    def _build_term_index(self):
        """
        Строим индекс определений за O(N_chunks × N_sentences_per_chunk),
        переиспользуя предвычисленные леммы предложений.
        """
        self.term_index: Dict[str, Dict] = {}

        # Предвычисляем леммы терминов
        term_lemmas_map = {}
        for term in self.terms:
            tl = [_parse_word(w)[0] for w in term.lower().split()]
            term_lemmas_map[term] = tl

        for term in self.terms:
            term_lemmas = term_lemmas_map[term]
            found = False

            for i, sent_list in enumerate(self._chunk_sentences):
                for sent_data in sent_list:
                    score = _score_sentence(term_lemmas, sent_data)
                    if score >= 0.5:
                        self.term_index[term] = {
                            'text': self.chunks[i],
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
                    self.term_index[term] = results[0]

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

        1. TF-IDF поиск (через _tfidf_search)
        2. Бустинг заголовков: query содержит название заголовка чанка → ×1.5
        3. Бустинг терминов: query совпадает с термином из terms.txt → ×1.3
        4. Сортировка по убыванию score, возврат топ-k
        """
        # 1. TF-IDF (берём с запасом, чтобы после бустинга не потерять релевантные)
        tfidf_results = self._tfidf_search(query, top_k=top_k * 2)

        query_lower = query.lower()
        detected_term = self._detect_term_in_query(query)

        results: List[SearchResult] = []
        for r in tfidf_results:
            chunk = self._chunks[r['chunk_id']]
            score = r['score']
            term_match: Optional[str] = None

            # 2. Бустинг заголовка
            if chunk.title.lower() in query_lower:
                score *= 1.5

            # 3. Бустинг термина
            if detected_term:
                score *= 1.3
                term_match = detected_term

            results.append(SearchResult(
                chunk=chunk,
                score=score,
                term_match=term_match,
            ))

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
