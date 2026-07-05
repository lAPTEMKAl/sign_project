"""
inference.py
Инференс одного изображения лучшей моделью + сохранение истории запусков
в SQLite (runs/history.db). Используется и из CLI, и из demo/app.py.
"""
import argparse
import datetime
import sqlite3
import time
from pathlib import Path

import torch
from PIL import Image

from datasets import build_transforms
from models import build_model

DB_PATH = Path("runs/history.db")


def init_db(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False — нужно для Streamlit (разные потоки)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            image_name TEXT,
            arch TEXT,
            predicted_class TEXT,
            confidence REAL,
            inference_ms REAL
        )
    """)
    conn.commit()
    return conn


def log_run(conn, image_name, arch, predicted_class, confidence, inference_ms):
    conn.execute(
        "INSERT INTO history (timestamp, image_name, arch, predicted_class, confidence, inference_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.datetime.now().isoformat(), image_name, arch, predicted_class, confidence, inference_ms),
    )
    conn.commit()


class Predictor:
    """Загружает чекпойнт один раз, переиспользуется в demo/app.py."""

    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.arch = ckpt["arch"]
        self.classes = ckpt["classes"]
        self.img_size = ckpt["img_size"]
        self.model = build_model(self.arch, len(self.classes), pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.transform = build_transforms(self.img_size, train=False)

    def predict(self, image: Image.Image):
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        start = time.time()
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
        inference_ms = (time.time() - start) * 1000
        conf, idx = probs.max(dim=0)
        topk_probs, topk_idx = probs.topk(min(3, len(self.classes)))
        topk = [(self.classes[i], float(p)) for p, i in zip(topk_probs, topk_idx)]
        return {
            "predicted_class": self.classes[idx.item()],
            "confidence": float(conf),
            "inference_ms": inference_ms,
            "topk": topk,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/best_model.pt")
    ap.add_argument("--image", type=str, required=True)
    args = ap.parse_args()

    predictor = Predictor(args.model)
    image = Image.open(args.image)
    result = predictor.predict(image)

    conn = init_db()
    log_run(conn, Path(args.image).name, predictor.arch, result["predicted_class"],
            result["confidence"], result["inference_ms"])

    print(f"Класс: {result['predicted_class']} | уверенность: {result['confidence']:.3f} | "
          f"время: {result['inference_ms']:.1f} мс")
    print("Топ-3:", result["topk"])


if __name__ == "__main__":
    main()
