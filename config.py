import os
import torch
import sys

# =====================================================
# 🛠️ ГЛАВНЫЕ НАСТРОЙКИ
# =====================================================

# 1. Настройка путей (САМОЕ ВАЖНОЕ ДЛЯ ДОКЕРА)
# Логика: Если мы в Докере, переменная окружения DATA_PATH будет равна "/data".
# Если мы запускаем локально в PyCharm, переменной нет, и берется "D:\SoundUniverse_ML".
DEFAULT_LOCAL_PATH = r"D:\SoundUniverse_ML"
BASE_DATA_DIR = os.environ.get("DATA_PATH", DEFAULT_LOCAL_PATH)

# Проверка, чтобы ты сразу понял, если путь не найден
if not os.path.exists(BASE_DATA_DIR):
    print(f"⚠️ ВНИМАНИЕ: Папка с данными не найдена по пути: {BASE_DATA_DIR}")
    # В Докере папка создастся сама при монтировании, но локально она должна быть.

# 2. Параметры Аудио
SAMPLE_RATE = 22050     # Частота дискретизации
SEGMENT_DURATION = 3    # Время одного сегмента
N_MELS = 64             # Высота картинки (пиксели)
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION

# 3. Параметры Обучения
BATCH_SIZE = 16         # Размер пачки данных
LEARNING_RATE = 0.001   # Размер шага для корректировки весов
EPOCHS = 30             # Количество эпох для обучения
WEIGHT_DECAY = 1e-3     # Штраф за высокое значение весов

# =====================================================
# ⚙️ АВТОМАТИЧЕСКАЯ НАСТРОЙКА (НЕ ТРОГАТЬ)
# =====================================================

# Теперь все папки строятся от BASE_DATA_DIR.
# Это позволяет коду работать и на Диске D, и в Докере в папке /data

# Пути к исходным и обработанным данным
RAW_DATASET_DIR = os.path.join(BASE_DATA_DIR, "raw_dataset") # Твой исходный датасет
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "processed_data")
TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "train")
VAL_DATA_PATH = os.path.join(PROCESSED_DIR, "val")

# Пути к моделям и логам
MODELS_DIR = os.path.join(BASE_DATA_DIR, "models")
LOG_DIR = os.path.join(BASE_DATA_DIR, "logs")
GRAPHICS_DIR = os.path.join(BASE_DATA_DIR, "graphics") # Добавил, раз она есть на скрине

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")

# Создание папок, если их нет (чтобы скрипт не падал)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Настройка FFMPEG
# Логика: В Windows мы указываем путь. В Linux (Docker) FFMPEG ставится в систему, путь добавлять не надо.
if sys.platform == "win32":
    LOCAL_FFMPEG = r"C:\ffmpeg\bin"
    if os.path.exists(LOCAL_FFMPEG):
        os.environ["PATH"] += os.pathsep + LOCAL_FFMPEG
    else:
        print(f"⚠️ ВНИМАНИЕ: FFMPEG не найден на Windows по пути {LOCAL_FFMPEG}")
else:
    # Для Linux/Docker ничего делать не нужно, там ffmpeg будет доступен по команде 'ffmpeg'
    pass

# Выбор устройства (GPU или CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Project is running on: {DEVICE}")
print(f"Data directory is set to: {BASE_DATA_DIR}")