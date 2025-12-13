import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import datetime
import logging
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ИМПОРТИРУЕМ НАСТРОЙКИ
import config

# Настройка логов
log_filename = f"train_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    filename=os.path.join(config.LOG_DIR, log_filename),
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    encoding='utf-8'
)


def log_print(msg):
    print(msg)
    logging.info(msg)


class SoundDataset(Dataset):
    def __init__(self, root_dir, is_train=False):
        self.root_dir = root_dir
        self.is_train = is_train
        self.files = []
        self.labels = []
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Папка {root_dir} не найдена!")
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(cls_folder):
                if file_name.endswith('.npy'):
                    self.files.append(os.path.join(cls_folder, file_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            spectrogram = np.load(self.files[idx])
            # Аугментация
            if self.is_train:
                # 1. Сдвиг
                if np.random.random() > 0.5:
                    shift = np.random.randint(-10, 10)
                    spectrogram = np.roll(spectrogram, shift, axis=1)
                # 2. Шум
                noise = np.random.normal(0, 0.01, spectrogram.shape)
                spectrogram = spectrogram + noise
                # 3. Frequency Masking
                if np.random.random() > 0.3:
                    h = spectrogram.shape[0]
                    mask_w = np.random.randint(5, 20)
                    mask_start = np.random.randint(0, h - mask_w)
                    spectrogram[mask_start:mask_start + mask_w, :] = 0.0

            tensor = torch.from_numpy(spectrogram).float().unsqueeze(0)
            return tensor, self.labels[idx]
        except:
            return torch.zeros(1, config.N_MELS, 130), 0


class MusicCNN(nn.Module):
    def __init__(self, num_classes):
        super(MusicCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train_epoch(model, loader, criterion, optimizer, epoch, epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start = time.time()
    loop = tqdm(loader, desc=f"Эпоха {epoch + 1}/{epochs}", leave=False)
    for data, targets in loop:
        data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)
        optimizer.zero_grad()
        preds = model(data)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(preds, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        loop.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader), 100 * correct / total, time.time() - start


def val_epoch(model, loader, criterion, epoch, epochs):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    start = time.time()
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)
            preds = model(data)
            loss = criterion(preds, targets)
            running_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return running_loss / len(loader), 100 * correct / total, time.time() - start


def main():
    log_print(f"⚙️ Устройство: {config.DEVICE}")
    log_print(f"🚀 Параметры: Кол-во эпох={config.EPOCHS}, BATCH={config.BATCH_SIZE}, LR={config.LEARNING_RATE}")

    train_ds = SoundDataset(config.TRAIN_DATA_PATH, is_train=True)
    val_ds = SoundDataset(config.VAL_DATA_PATH, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    log_print(f"📂 Данных на обучении: {len(train_ds)} | Данных на валидации: {len(val_ds)} | Жанров найдено: {len(train_ds.classes)}")

    model = MusicCNN(len(train_ds.classes)).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        t_loss, t_acc, t_time = train_epoch(model, train_loader, criterion, optimizer, epoch, config.EPOCHS)
        v_loss, v_acc, v_time = val_epoch(model, val_loader, criterion, epoch, config.EPOCHS)
        scheduler.step(v_loss)

        log_print(f"Эпоха {epoch + 1} | Обучение: {t_acc:.2f}% ({t_time:.0f}s) | Валидация: {v_acc:.2f}% ({v_time:.0f}s)")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            log_print(f"   💾 Новый Рекорд! ({best_acc:.2f}%)")

    log_print("🏁 Обучение завершено.")


if __name__ == "__main__":
    main()