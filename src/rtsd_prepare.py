"""
rtsd_prepare.py
Конвертация ПОЛНОЙ версии RTSD (rtsd-frames + train_anno.json + val_anno.json)
в структуру ImageFolder (папка = класс) пригодную для обучения классификатора.

Формат аннотаций RTSD:
{
  "annotations": [
    {
      "id": 1,
      "image_id": 123,
      "category_id": 5,
      "bbox": [x, y, width, height]
    }, ...
  ],
  "images": [
    {"id": 123, "file_name": "rtsd-frames/000001/frame_000001.jpg"}, ...
  ],
  "categories": [
    {"id": 5, "name": "2.1"}, ...
  ]
}

Выход: data/processed/{train,val,test}/<class_name>/<crop>.jpg
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

MARGIN = 0.05   # добавляем 5% отступа вокруг bbox знака


def load_anno(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_id_maps(anno):
    images   = {img["id"]: img["file_name"] for img in anno["images"]}
    cats     = {cat["id"]: cat["name"]       for cat in anno["categories"]}
    return images, cats


def crop_and_save(raw_full: Path, anno_file: Path, out_dir: Path, split_name: str,
                  min_size: int = 20, margin: float = MARGIN):
    anno    = load_anno(anno_file)
    images, cats = build_id_maps(anno)

    saved = 0
    skipped = 0

    for ann in anno["annotations"]:
        img_rel  = images.get(ann["image_id"])
        cat_name = cats.get(ann["category_id"])
        if img_rel is None or cat_name is None:
            skipped += 1
            continue

        # sanitize class name (убираем / и пробелы для имени папки)
        safe_cls = cat_name.replace("/", "_").replace(" ", "_")

        img_path = raw_full / img_rel
        if not img_path.exists():
            skipped += 1
            continue

        x, y, w, h = ann["bbox"]
        if w < min_size or h < min_size:
            skipped += 1
            continue

        try:
            with Image.open(img_path) as im:
                W, H = im.size
                mx, my = w * margin, h * margin
                x1 = max(0, x - mx)
                y1 = max(0, y - my)
                x2 = min(W, x + w + mx)
                y2 = min(H, y + h + my)
                crop = im.crop((x1, y1, x2, y2)).convert("RGB")

                dst_dir = out_dir / split_name / safe_cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                crop.save(dst_dir / f"{ann['id']}.jpg", quality=92)
                saved += 1
        except Exception:
            skipped += 1

    print(f"  [{split_name}] сохранено: {saved}, пропущено: {skipped}")
    return saved


def make_test_split(processed: Path, test_ratio: float = 0.15, seed: int = 42):
    """Берём часть train и переносим в test."""
    random.seed(seed)
    train_dir = processed / "train"
    test_dir  = processed / "test"

    for cls_dir in sorted(train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        files = list(cls_dir.iterdir())
        random.shuffle(files)
        n_test = max(1, int(len(files) * test_ratio))
        (test_dir / cls_dir.name).mkdir(parents=True, exist_ok=True)
        for f in files[:n_test]:
            shutil.move(str(f), test_dir / cls_dir.name / f.name)

    print(f"  [test] выделен из train (ratio={test_ratio})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-full",    default="data/raw_full",
                    help="Папка с распакованным RTSD (содержит rtsd-frames и *.json)")
    ap.add_argument("--out",         default="data/processed")
    ap.add_argument("--test-ratio",  type=float, default=0.15)
    ap.add_argument("--min-size",    type=int,   default=20,
                    help="Минимальный размер стороны bbox (пикс.) — маленькие пропускаем")
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    raw_full  = Path(args.raw_full)
    out_dir   = Path(args.out)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    train_json = raw_full / "train_anno.json"
    val_json   = raw_full / "val_anno.json"

    assert train_json.exists(), f"Не найден {train_json}"
    assert val_json.exists(),   f"Не найден {val_json}"

    print("Вырезаем знаки из train кадров...")
    crop_and_save(raw_full, train_json, out_dir, "train", args.min_size)

    print("Вырезаем знаки из val кадров...")
    crop_and_save(raw_full, val_json,   out_dir, "val",   args.min_size)

    print("Выделяем test из train...")
    make_test_split(out_dir, args.test_ratio, args.seed)

    # Статистика
    for split in ("train", "val", "test"):
        sdir = out_dir / split
        if not sdir.exists():
            continue
        n_cls   = sum(1 for d in sdir.iterdir() if d.is_dir())
        n_files = sum(len(list(d.iterdir())) for d in sdir.iterdir() if d.is_dir())
        print(f"  {split}: {n_files} изображений, {n_cls} классов")

    print(f"\nГотово! Данные в {out_dir}")
    print("Следующий шаг:")
    print("  python src/run_all.py --data data/processed --out runs/experiment_1")


if __name__ == "__main__":
    main()
