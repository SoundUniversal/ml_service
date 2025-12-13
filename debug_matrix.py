import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from train import MusicCNN, SoundDataset
import config


def plot_confusion_matrix():
    print("⏳ Загрузка...")
    val_dataset = SoundDataset(config.VAL_DATA_PATH, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = MusicCNN(len(val_dataset.classes)).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    print("🔍 Анализ...")
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(config.DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("📊 Рисуем график...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=val_dataset.classes, yticklabels=val_dataset.classes)
    plt.title(f'Матрица ошибок')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_confusion_matrix()