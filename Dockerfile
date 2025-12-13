# 1. Базовый образ (Linux + Python 3.9)
FROM python:3.9-slim

# 2. Устанавливаем системные библиотеки (включая FFmpeg!)
# Это команда Linux, она сама скачает и поставит ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Рабочая папка внутри контейнера
WORKDIR /app

# 4. Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Копируем ВЕСЬ код проекта внутрь
COPY . .

# 6. Создаем папку для логов
RUN mkdir -p logs

# 7. Команда запуска.
CMD ["python", "app.py"]