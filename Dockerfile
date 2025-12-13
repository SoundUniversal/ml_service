FROM python:3.9-slim

# 2. Устанавливаем системные библиотеки (FFMPEG нужен для librosa!)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Настройка переменных окружения
# (запрещаем создавать .pyc файлы и буферизировать вывод)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Создаем рабочую папку внутри контейнера
WORKDIR /app

# 5. Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Копируем весь код проекта в контейнер
COPY . .

# 7. Открываем порт 8000 (так как у тебя app.py работает на нем)
EXPOSE 8000

# 8. Команда запуска по умолчанию (запускаем сервер)
CMD ["python", "app.py"]