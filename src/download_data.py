"""
download_data.py
Скачивание / подготовка датасета RTSD (Russian Traffic Sign Dataset).

RTSD распространяется через несколько источников (полный архив большой, ~30+ ГБ
с видео и полной разметкой детекции). Для учебного проекта по КЛАССИФИКАЦИИ
рекомендуется использовать уже нарезанные кропы знаков (classification subset),
которые встречаются на Kaggle под названиями вроде:
    - "Russian Traffic Sign Images Dataset" (RTSD classification crops)
    - "rtsd-r1" / "rtsd-r3" classification archives

Так как зеркала и точные ссылки периодически меняются, este скрипт:
  1) Если найден локальный архив (--archive) — распаковывает его.
  2) Иначе печатает пошаговую инструкцию, что и где скачать руками
     (Kaggle требует авторизации, поэтому автоматическая загрузка ненадёжна).

После любого варианта данные должны оказаться в структуре:

    data/raw/
        images/
            <sign_class_id_or_name>/
                img_0001.jpg
                img_0002.jpg
                ...
        (опционально) annotations.csv  # path,class,x1,y1,x2,y2 — если есть bbox-разметка

Если у вас исходно есть полные кадры с bbox-разметкой (формат RTSD full),
используйте --crop-from-bbox, чтобы автоматически вырезать знаки в нужную структуру.
"""
import argparse
import csv
import os
import shutil
import zipfile
from pathlib import Path

from PIL import Image


INSTRUCTIONS = """
=========================================================================
RTSD не имеет единого "официального" URL для прямого wget — это требует
авторизации (Kaggle) или ручного скачивания. Шаги:

1. Зайти на Kaggle и найти один из датасетов:
   - "Russian Traffic Sign Images Dataset (RTSD)"
   - "RTSD - Russian traffic sign dataset"
   Поиск: https://www.kaggle.com/search?q=RTSD+traffic+sign

2. Скачать архив (классификационную версию, где знаки уже нарезаны по классам,
   либо полную с bbox-разметкой).

3. Если скачали classification-версию (папки = классы):
   python src/download_data.py --archive /path/to/rtsd.zip --target data/raw

4. Если скачали full-версию (кадры + bbox CSV/JSON):
   python src/download_data.py --archive /path/to/rtsd_full.zip --target data/raw_full
   python src/download_data.py --crop-from-bbox --raw-full data/raw_full \\
       --annotations data/raw_full/annotations.csv --target data/raw

5. Альтернатива (если RTSD недоступен в вашем регионе или возникли проблемы
   с доступом): использовать GTSRB как запасной вариант — структура датасета
   эквивалентна (ImageFolder), переключение не требует правки остального кода.
   https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
=========================================================================
"""


def extract_archive(archive_path: str, target: str):
    target_p = Path(target)
    target_p.mkdir(parents=True, exist_ok=True)
    print(f"Распаковка {archive_path} -> {target}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_p)
    print("Готово.")


def crop_from_bbox(raw_full: str, annotations_csv: str, target: str, margin: float = 0.05):
    """
    Ожидаемый формат annotations.csv: filename,class,x1,y1,x2,y2
    (под формат RTSD full при необходимости адаптируйте парсинг ниже).
    """
    raw_full_p = Path(raw_full)
    target_p = Path(target) / "images"
    target_p.mkdir(parents=True, exist_ok=True)

    with open(annotations_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            img_path = raw_full_p / row["filename"]
            if not img_path.exists():
                continue
            cls = row["class"]
            x1, y1, x2, y2 = (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )
            with Image.open(img_path) as im:
                w, h = im.size
                mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
                x1, y1 = max(0, x1 - mw), max(0, y1 - mh)
                x2, y2 = min(w, x2 + mw), min(h, y2 + mh)
                crop = im.crop((x1, y1, x2, y2))
                out_dir = target_p / str(cls)
                out_dir.mkdir(parents=True, exist_ok=True)
                crop.save(out_dir / f"{Path(row['filename']).stem}_{i}.jpg")
    print(f"Кропы сохранены в {target_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=str, default=None, help="Путь к zip-архиву датасета")
    ap.add_argument("--target", type=str, default="data/raw")
    ap.add_argument("--crop-from-bbox", action="store_true")
    ap.add_argument("--raw-full", type=str, default=None)
    ap.add_argument("--annotations", type=str, default=None)
    args = ap.parse_args()

    if args.crop_from_bbox:
        assert args.raw_full and args.annotations, "Нужны --raw-full и --annotations"
        crop_from_bbox(args.raw_full, args.annotations, args.target)
        return

    if args.archive:
        extract_archive(args.archive, args.target)
        return

    print(INSTRUCTIONS)


if __name__ == "__main__":
    main()
