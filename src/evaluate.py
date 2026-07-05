"""
evaluate.py
Подробная оценка одной обученной модели на test-выборке: accuracy, precision,
recall, F1 (macro и per-class), confusion matrix (PNG), а также сохранение
минимум 3 успешных и 3 ошибочных примеров (требование методички, раздел 6),
с акцентом на ошибки между похожими классами знаков.
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support)

from datasets import build_dataloaders
from models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Путь к .pt чекпойнту")
    ap.add_argument("--data", type=str, default="data/processed")
    ap.add_argument("--out", type=str, default="runs/eval")
    ap.add_argument("--n-examples", type=int, default=6, help="Сколько успешных/ошибочных примеров сохранить")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    arch, classes, img_size = checkpoint["arch"], checkpoint["classes"], checkpoint["img_size"]

    model = build_model(arch, len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, _, test_loader, ds_classes = build_dataloaders(args.data, img_size=img_size, batch_size=32)
    assert ds_classes == classes, "Классы чекпойнта не совпадают с текущим датасетом"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    correct_dir, wrong_dir = out_dir / "examples_correct", out_dir / "examples_wrong"
    correct_dir.mkdir(exist_ok=True)
    wrong_dir.mkdir(exist_ok=True)

    all_preds, all_labels, all_confidences = [], [], []
    saved_correct = saved_wrong = 0
    sample_idx = 0

    test_dataset = test_loader.dataset
    with torch.no_grad():
        for images, labels in test_loader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            for i in range(images.size(0)):
                pred, label, conf = preds[i].item(), labels[i].item(), confs[i].item()
                all_preds.append(pred)
                all_labels.append(label)
                all_confidences.append(conf)

                src_path, _ = test_dataset.samples[sample_idx]
                if pred == label and saved_correct < args.n_examples:
                    shutil.copy2(src_path, correct_dir / f"{classes[label]}_conf{conf:.2f}_{Path(src_path).name}")
                    saved_correct += 1
                elif pred != label and saved_wrong < args.n_examples:
                    shutil.copy2(src_path, wrong_dir /
                                  f"true-{classes[label]}_pred-{classes[pred]}_conf{conf:.2f}_{Path(src_path).name}")
                    saved_wrong += 1
                sample_idx += 1

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": acc, "precision_macro": precision, "recall_macro": recall, "f1_macro": f1,
        "mean_confidence": float(np.mean(all_confidences)),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Confusion matrix (полная, плюс top-confused pairs отдельно — полезно для похожих знаков)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(max(8, len(classes) * 0.3), max(6, len(classes) * 0.3)))
    sns.heatmap(cm, cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=True)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title(f"Confusion matrix — {arch}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Топ-10 наиболее путаемых пар классов (для анализа ошибок между похожими знаками)
    cm_no_diag = cm.copy().astype(float)
    np.fill_diagonal(cm_no_diag, 0)
    confused_pairs = []
    flat_idx = np.argsort(cm_no_diag, axis=None)[::-1][:10]
    for idx in flat_idx:
        i, j = np.unravel_index(idx, cm_no_diag.shape)
        if cm_no_diag[i, j] > 0:
            confused_pairs.append({"true": classes[i], "predicted": classes[j], "count": int(cm_no_diag[i, j])})
    with open(out_dir / "top_confused_pairs.json", "w", encoding="utf-8") as f:
        json.dump(confused_pairs, f, ensure_ascii=False, indent=2)

    print("Метрики:", json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Топ путаемых пар классов:", confused_pairs[:5])
    print(f"Примеры сохранены в {correct_dir} и {wrong_dir}")


if __name__ == "__main__":
    main()
