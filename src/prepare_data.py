"""
prepare_data.py
Проверка разметки, разбиение на train/val/test, подготовка структуры ImageFolder.

Вход:  data/raw/images/<class_id>/*.jpg
Выход: data/processed/{train,val,test}/<class_id>/*.jpg
       data/processed/class_stats.csv  — статистика по классам (для отчёта)
       data/processed/split_seed.txt    — зафиксированный seed разбиения

Соответствует методическим рекомендациям (разделы 3 и 6):
  - фиксированный seed разбиения,
  - проверка баланса классов,
  - явное train/val/test без утечки данных (один и тот же файл не попадает
    в несколько выборок).
"""
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def collect_classes(raw_images_dir: Path):
    class_to_files = defaultdict(list)
    for class_dir in sorted(raw_images_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                class_to_files[class_dir.name].append(f)
    return class_to_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="data/raw")
    ap.add_argument("--out", type=str, default="data/processed")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--test-split", type=float, default=0.15)
    ap.add_argument("--min-per-class", type=int, default=20,
                     help="Классы с меньшим числом изображений будут исключены и залогированы")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-validation", action="store_true",
                     help="Не проверять каждое изображение на повреждённость (быстрее, но рискованнее)")
    args = ap.parse_args()

    random.seed(args.seed)
    raw_images_dir = Path(args.raw) / "images"
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    class_to_files = collect_classes(raw_images_dir)
    print(f"Найдено классов: {len(class_to_files)}")

    stats_rows = []
    excluded = []
    total_train = total_val = total_test = 0

    for cls, files in class_to_files.items():
        if not args.skip_validation:
            files = [f for f in files if is_valid_image(f)]

        if len(files) < args.min_per_class:
            excluded.append((cls, len(files)))
            continue

        random.shuffle(files)
        n = len(files)
        n_val = max(1, int(n * args.val_split))
        n_test = max(1, int(n * args.test_split))
        n_train = n - n_val - n_test

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for split_name, split_files in (("train", train_files), ("val", val_files), ("test", test_files)):
            dst_dir = out_dir / split_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dst_dir / f.name)

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)
        stats_rows.append({
            "class": cls, "total": n,
            "train": len(train_files), "val": len(val_files), "test": len(test_files),
        })

    with open(out_dir / "class_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "total", "train", "val", "test"])
        writer.writeheader()
        writer.writerows(stats_rows)

    with open(out_dir / "split_seed.txt", "w") as f:
        f.write(f"seed={args.seed}\nval_split={args.val_split}\ntest_split={args.test_split}\n")

    print(f"Итог: train={total_train}, val={total_val}, test={total_test}")
    print(f"Классов оставлено: {len(stats_rows)}, исключено (мало примеров): {len(excluded)}")
    if excluded:
        print("Исключённые классы (class, count):", excluded[:20], "..." if len(excluded) > 20 else "")
    print(f"Статистика сохранена в {out_dir / 'class_stats.csv'}")


if __name__ == "__main__":
    main()
