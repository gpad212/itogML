# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Скопируем файлы сначала requirements, чтобы закэшировать зависимости
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем NLTK ресурсы в известное место
RUN python -m nltk.downloader wordnet omw-1.4 punkt stopwords -d /usr/local/share/nltk_data

# Копируем приложение
COPY . /app

# Убедимся, что путь NLTK доступен
ENV NLTK_DATA=/usr/local/share/nltk_data
ENV MODEL_DIR=/app/model
ENV PYTHONPATH=/app

RUN mkdir -p /app/tmp /app/output /app/model

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
