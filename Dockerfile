# RAG-система по теории графов — Docker образ
# Python бэкенд (Flask + sklearn) + статика фронтенда через Flask

FROM python:3.11-slim

WORKDIR /app

# Системные зависимости минимальные
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/

# Flask будет раздавать и API и статику
COPY serve.py ./serve.py

EXPOSE 5000

# Запуск через gunicorn для стабильности
RUN pip install --no-cache-dir gunicorn

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "serve:app"]
