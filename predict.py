import torch
import librosa
import numpy as np
import os
import torch.nn.functional as F
from train import MusicCNN
import config


def get_genres(data_dir):
    return sorted(os.listdir(data_dir))


def format_verdict(probs, indices, genres):
    p1, p2 = probs[0], probs[1]
    g1, g2 = genres[indices[0]], genres[indices[1]]
    if p1 > 0.75: return g1.upper()
    if p1 < 0.25: return "НЕИЗВЕСТНО"
    if (p1 - p2) < 0.15: return f"{g1.title()} / {g2.title()}"
    return g1.upper()


def analyze_track(audio_path):
    if not os.path.exists(config.TRAIN_DATA_PATH): return {"error": "Нет датасета"}
    genres = get_genres(config.TRAIN_DATA_PATH)

    model = MusicCNN(len(genres)).to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
        model.eval()
    except:
        return {"error": "Модель не найдена"}

    try:
        signal, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)
    except Exception as e:
        return {"error": f"Ошибка файла: {e}"}

    num_segments = int(len(signal) / config.SAMPLES_PER_SEGMENT)
    if num_segments == 0: return {"error": "Трек слишком короткий"}

    batch = []
    for s in range(num_segments):
        start = s * config.SAMPLES_PER_SEGMENT
        segment = signal[int(start):int(start + config.SAMPLES_PER_SEGMENT)]

        melspec = librosa.feature.melspectrogram(y=segment, sr=config.SAMPLE_RATE, n_mels=config.N_MELS)
        melspec = (librosa.power_to_db(melspec, ref=np.max) + 80) / 80
        batch.append(torch.from_numpy(melspec).float().unsqueeze(0))

    with torch.no_grad():
        outputs = model(torch.stack(batch).to(config.DEVICE))
        probs = F.softmax(outputs, dim=1).mean(dim=0).cpu().numpy()

    sorted_idx = probs.argsort()[::-1]
    sorted_probs = probs[sorted_idx]

    verdict = format_verdict(sorted_probs, sorted_idx, genres)

    return {
        "verdict": verdict,
        "primary_genre": genres[sorted_idx[0]],
        "confidence": round(float(sorted_probs[0] * 100), 2),
        "details": {genres[i]: round(float(probs[i] * 100), 1) for i in sorted_idx[:3]}
    }


if __name__ == "__main__":
    path = input("Путь к mp3: ").strip('"')
    if os.path.exists(path):
        print(analyze_track(path))