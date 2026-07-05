"""
train.py
Обучение/дообучение одной архитектуры. Логирует метрики по эпохам,
сохраняет лучший чекпойнт по val accuracy, измеряет среднее время инференса
одного изображения и размер модели — всё это требуется для таблицы сравнения
(методичка, раздел 6.1).
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets import build_dataloaders
from models import build_model, model_size_mb


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def measure_inference_time(model, device, img_size, n_runs=50):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_runs):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
    return (elapsed / n_runs) * 1000  # мс на изображение


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, required=True)
    ap.add_argument("--data", type=str, default="data/processed")
    ap.add_argument("--config", type=str, default="configs/models.yaml")
    ap.add_argument("--out", type=str, default="runs/single_run")
    ap.add_argument("--epochs", type=int, default=None, help="Переопределить значение из конфига")
    ap.add_argument("--img-size", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    common = cfg["common"]
    model_cfg = cfg["models"][args.arch]

    img_size = args.img_size or common["img_size"]
    epochs = args.epochs or model_cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Архитектура: {args.arch} | устройство: {device} | epochs={epochs} | img_size={img_size}")

    train_loader, val_loader, test_loader, classes = build_dataloaders(
        args.data, img_size=img_size, batch_size=common["batch_size"], num_workers=common["num_workers"])
    num_classes = len(classes)

    model = build_model(args.arch, num_classes, pretrained=model_cfg.get("pretrained", True)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=common.get("label_smoothing", 0.0))
    optimizer = AdamW(model.parameters(), lr=model_cfg["lr"], weight_decay=common.get("weight_decay", 0.0))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir = Path(args.out) / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_acc = 0.0
    patience = common.get("early_stopping_patience", 5)
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"[{args.arch}] epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "classes": classes, "arch": args.arch,
                        "img_size": img_size}, out_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping на эпохе {epoch+1}")
                break

    # Финальная оценка ОДИН раз на test (после выбора по val, как требует методичка)
    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

    avg_inference_ms = measure_inference_time(model, device, img_size)
    size_mb = model_size_mb(model)

    summary = {
        "arch": args.arch,
        "device": str(device),
        "img_size": img_size,
        "epochs_run": len(history),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "avg_inference_ms": avg_inference_ms,
        "model_size_mb": size_mb,
        "num_classes": num_classes,
        "checkpoint": str(out_dir / "best.pt"),
    }

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Итог:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
