import os
import sys
import shutil
import librosa
import numpy as np
import logging
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ИМПОРТИРУЕМ НАСТРОЙКИ
import config

# Настройка логов
log_filename = f"preprocess_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    filename=os.path.join(config.LOG_DIR, log_filename),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def process_and_save(files, source_genre_path, output_genre_path, genre_name, skipped_list):
    count = 0
    # tqdm пишет в stdout, чтобы не засорять логи
    for f in tqdm(files, desc=f"Жанр: {genre_name}", file=sys.stdout):
        if not f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
            continue

        file_path = os.path.join(source_genre_path, f)
        try:
            # Читаем длительность
            try:
                full_duration = librosa.get_duration(path=file_path)
            except Exception:
                raise ValueError("Не удалось прочитать заголовок")

            # Логика нарезки (45 или 30)
            if full_duration >= 45:
                crop_size = 45
            elif full_duration >= 30:
                crop_size = 30
            else:
                crop_size = full_duration

            if full_duration < config.SEGMENT_DURATION:
                raise ValueError(f"Короче {config.SEGMENT_DURATION} сек")

            center = full_duration / 2
            start_time = max(0, center - (crop_size / 2))

            signal, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, offset=start_time, duration=crop_size)

            num_segments = int(len(signal) / config.SAMPLES_PER_SEGMENT)
            file_code = os.path.splitext(f)[0].replace(" ", "_")

            if num_segments == 0: raise ValueError("0 сегментов")

            for s in range(num_segments):
                start = int(config.SAMPLES_PER_SEGMENT * s)
                finish = int(start + config.SAMPLES_PER_SEGMENT)
                segment = signal[start:finish]

                if len(segment) != int(config.SAMPLES_PER_SEGMENT): continue

                melspec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=config.N_MELS)
                melspec_db = librosa.power_to_db(melspec, ref=np.max)
                melspec_norm = (melspec_db + 80) / 80

                save_name = f"{genre_name}_{file_code}_seg{s}.npy"
                np.save(os.path.join(output_genre_path, save_name), melspec_norm)
                count += 1

        except Exception as e:
            error_msg = f"{genre_name} | {f} -> {str(e)}"
            skipped_list.append(error_msg)
            logging.warning(error_msg)

    return count


def prepare_dataset():
    logging.info("🚀 СТАРТ ПОДГОТОВКИ")
    logging.info(f"Параметры: SR={config.SAMPLE_RATE}, MELS={config.N_MELS}")

    source_path = config.RAW_DATASET_DIR

    if os.path.exists(config.PROCESSED_DIR):
        logging.info("🧹 Очистка старых данных...")
        shutil.rmtree(config.PROCESSED_DIR)

    total_skipped = []
    total_segments = 0

    try:
        # Проверяем наличие папки с сырыми данными
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Папка {source_path} не найдена!")

        genres = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
    except FileNotFoundError as e:
        logging.error(f"❌ Ошибка: {e}")
        print(f"\n🛑 ОШИБКА: {e}")
        print(f"Проверь, что по пути {source_path} лежат папки с музыкой.")
        return

    logging.info(f"Найдено жанров: {len(genres)}")

    for genre in genres:
        source_genre_path = os.path.join(source_path, genre)
        all_files = os.listdir(source_genre_path)

        if len(all_files) < 2:
            train, val = all_files, []
        else:
            train, val = train_test_split(all_files, test_size=0.15, random_state=42)

        out_train = os.path.join(config.TRAIN_DATA_PATH, genre)
        out_val = os.path.join(config.VAL_DATA_PATH, genre)
        os.makedirs(out_train, exist_ok=True)
        os.makedirs(out_val, exist_ok=True)

        c1 = process_and_save(train, source_genre_path, out_train, genre, total_skipped)
        c2 = process_and_save(val, source_genre_path, out_val, genre, total_skipped)
        total_segments += (c1 + c2)

    logging.info(f"🏁 Готово. Сегментов: {total_segments}. Ошибок: {len(total_skipped)}")
    if len(total_skipped) > 0:
        with open("skipped_log.txt", "w", encoding="utf-8") as f:
            for e in total_skipped: f.write(e + "\n")
        logging.info("Лог ошибок сохранен в skipped_log.txt")


if __name__ == "__main__":
    prepare_dataset()