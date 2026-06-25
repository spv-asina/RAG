"""
llm.py — HTTP-клиент к llama.cpp серверу (Qwen 2.5 7B Q4_K_M).

Назначение:
  Улучшение формулировок вопросов и ответов, генерация дистракторов для тестов.
  Все методы имеют graceful degradation — при недоступности LLM возвращается исходный текст.

Правила:
  - ❌ НЕ может принимать решения (правильно/неправильно)
  - ❌ НЕ может менять смысл исходного текста
  - ✅ Может улучшать формулировки (делать более связными, грамматически правильными)
  - ✅ Может генерировать дистракторы для тестов (неверные, но правдоподобные варианты)
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    base_url: str = "http://llm-server:8080"
    timeout: int = 30
    enabled: bool = True
    max_retries: int = 2


class LLMService:
    """
    Клиент для llama.cpp server (или совместимого OpenAI-like API).
    Все методы возвращают исходные данные при недоступности LLM.
    """

    # Кэш доступности: проверка не чаще раза в HEALTH_CACHE_TTL секунд
    HEALTH_CACHE_TTL: int = 30

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._available = False
        self._last_health_check: float = 0.0

    def is_available(self) -> bool:
        """Проверить доступность LLM-сервера с кэшированием на 30 секунд."""
        if not self.config.enabled:
            return False

        now = time.time()
        if now - self._last_health_check < self.HEALTH_CACHE_TTL:
            return self._available

        self._last_health_check = now
        try:
            r = requests.get(
                f"{self.config.base_url}/health",
                timeout=2,
            )
            self._available = r.status_code == 200
        except requests.RequestException:
            self._available = False

        if not self._available:
            logger.warning("LLM-сервер недоступен")
        return self._available

    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """
        Вызвать LLM через OpenAI-compatible API (llama.cpp server).
        Возвращает текст ответа или None при ошибке.
        """
        if not self.is_available():
            logger.warning("LLM недоступен, пропускаем вызов")
            return None

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["\n\n", "---"],
        }

        if system_prompt:
            payload["system"] = system_prompt

        for attempt in range(self.config.max_retries):
            try:
                r = requests.post(
                    f"{self.config.base_url}/v1/completions",
                    json=payload,
                    timeout=self.config.timeout,
                )
                if r.status_code == 200:
                    data = r.json()
                    return data.get("choices", [{}])[0].get("text", "").strip()
                else:
                    self._available = False
                    logger.warning(
                        "LLM вернул %d: %s", r.status_code, r.text[:200]
                    )
            except requests.RequestException as e:
                self._available = False
                logger.warning("Попытка %d: ошибка LLM: %s", attempt + 1, e)

        return None

    def _validate_meaning_preserved(
        self, original: str, improved: str, key_terms: List[str]
    ) -> bool:
        """
        Проверить, что улучшенная версия содержит все ключевые термины
        и не искажает смысл (нет отрицаний рядом с терминами).
        Если нет — логировать и вернуть original.
        """
        improved_lower = improved.lower()
        missing = [t for t in key_terms if t.lower() not in improved_lower]
        if missing:
            logger.warning("LLM удалил ключевые термины: %s", missing)
            return False

        # Проверка на отрицания рядом с терминами (изменение смысла)
        for term in key_terms:
            neg_pattern = r'(?:не |не |отрицается|не является)\s*' + re.escape(term.lower())
            if re.search(neg_pattern, improved_lower):
                logger.warning(
                    "LLM изменил смысл: отрицание рядом с термином '%s'",
                    term,
                )
                return False

        return True

    def improve_question(self, question_text: str, context: str = "") -> str:
        """
        Улучшить формулировку вопроса, не меняя смысл.

        Args:
            question_text: Исходный текст вопроса
            context: Дополнительный контекст (определение термина)

        Returns:
            Улучшенный текст вопроса или исходный, если LLM недоступен
        """
        if not self.is_available():
            return question_text

        system = "Ты — ассистент, улучшающий формулировки учебных вопросов."
        context_part = context[:300] if context else "нет"
        prompt = (
            f"Улучши формулировку вопроса, сделав её более чёткой и грамматически правильной.\n"
            f"НЕ меняй смысл вопроса. НЕ добавляй новой информации.\n"
            f"Если вопрос уже сформулирован хорошо — верни как есть.\n\n"
            f"Контекст: {context_part}\n\n"
            f"Вопрос: {question_text}\n\n"
            f"Улучшенный вопрос:"
        )

        result = self._call_llm(
            prompt, system_prompt=system, temperature=0.2, max_tokens=200
        )

        if result:
            # Извлекаем значимые термины из исходного вопроса (слова длиннее 3 символов)
            key_terms = list(
                set(re.findall(r"[а-яёА-ЯЁa-zA-Z]{4,}", question_text.lower()))
            )
            if self._validate_meaning_preserved(question_text, result, key_terms):
                return result.strip()

        return question_text

    def improve_answer(self, answer_text: str, context: str = "") -> str:
        """
        Сделать текстовый ответ более связным и читаемым, не добавляя информации.

        Args:
            answer_text: Сырой ответ из учебника (чанк текста)
            context: Исходный вопрос (для контекста)

        Returns:
            Улучшенный текст
        """
        if not self.is_available():
            return answer_text

        system = "Ты — ассистент, улучшающий читаемость текста."
        prompt = (
            f"Перепиши следующий текст более связно, но НЕ добавляй новой информации.\n"
            f"Сохрани все термины, определения и факты в точности.\n"
            f"Улучши читаемость: разбей на предложения, исправь грамматику.\n\n"
            f"Исходный текст:\n{answer_text[:500]}\n\n"
            f"Улучшенный текст:"
        )

        result = self._call_llm(
            prompt, system_prompt=system, temperature=0.2, max_tokens=500
        )

        if result:
            # Извлекаем ключевые термины (слова длиннее 4 символов)
            key_terms = list(
                set(re.findall(r"[а-яёА-ЯЁa-zA-Z]{4,}", answer_text.lower()))
            )
            if self._validate_meaning_preserved(answer_text, result, key_terms):
                return result.strip()

        return answer_text

    def generate_distractors(
        self, term: str, definition: str, count: int = 3
    ) -> List[str]:
        """
        Сгенерировать неверные, но правдоподобные варианты для теста.

        Args:
            term: Правильный термин
            definition: Правильное определение
            count: Сколько дистракторов нужно

        Returns:
            Список неверных вариантов (или пустой список при ошибке)
        """
        if not self.is_available():
            return []

        system = "Ты — генератор учебных тестов."
        prompt = (
            f"Придумай {count} неправильных, но правдоподобных определений для термина «{term}».\n"
            f"Правильное определение: {definition[:200]}\n\n"
            f"Неправильные определения (каждое с новой строки, без нумерации):\n"
            f"1. "
        )

        result = self._call_llm(
            prompt, system_prompt=system, temperature=0.7, max_tokens=200
        )

        if result:
            lines = [
                line.strip().lstrip("0123456789.-) ")
                for line in result.split("\n")
                if line.strip()
            ]
            # Проверяем, что дистракторы не совпадают с правильным определением
            filtered = []
            for line in lines:
                if term.lower() not in line.lower()[:50]:
                    filtered.append(line[:150])
                if len(filtered) >= count:
                    break
            return filtered

        return []
