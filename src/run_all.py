import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ARCH_NAMES = ["resnet18", "mobilenet_v3_small", "efficientnet_b0", "densenet121", "vit_b_16"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed")
    ap.add_argument("--config", type=str, default="configs/models.yaml")
    ap.add_argument("--out", type=str, default="runs/experiment_1")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--archs", nargs="+", default=ARCH_NAMES)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Путь к train.py всегда рядом с run_all.py
    train_script = Path(__file__).parent / "train.py"

    summaries = []
    for arch in args.archs:
        print(f"\n{'='*60}\nОбучение архитектуры: {arch}\n{'='*60}")
        cmd = [sys.executable, str(train_script),
               "--arch", arch,
               "--data", args.data,
               "--config", args.config,
               "--out", str(out_dir)]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        if args.img_size:
            cmd += ["--img-size", str(args.img_size)]
        subprocess.run(cmd, check=True)

        summary_path = out_dir / arch / "summary.json"
        with open(summary_path, encoding="utf-8") as f:
            summaries.append(json.load(f))

    df = pd.DataFrame(summaries)
    df = df.sort_values("test_acc", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "comparison_table.csv", index=False, encoding="utf-8-sig")

    md_lines = [
        "| Модель | Input size | Эпохи | Test accuracy | Время/кадр (мс) | Размер (МБ) |",
        "|---|---|---|---|---|---|",
    ]
    for row in summaries:
        md_lines.append(
            f"| {row['arch']} | {row['img_size']}x{row['img_size']} | {row['epochs_run']} | "
            f"{row['test_acc']:.4f} | {row['avg_inference_ms']:.2f} | {row['model_size_mb']:.1f} |"
        )
    (out_dir / "comparison_table.md").write_text("\n".join(md_lines), encoding="utf-8")

    best = df.iloc[0]
    Path("models").mkdir(exist_ok=True)
    import shutil
    shutil.copy2(best["checkpoint"], "models/best_model.pt")

    print("\nСравнительная таблица:")
    print(df[["arch", "test_acc", "avg_inference_ms", "model_size_mb"]].to_string(index=False))
    print(f"\nЛучшая модель: {best['arch']} (acc={best['test_acc']:.4f})")
    print("Скопирована в models/best_model.pt")


if __name__ == "__main__":
    main()
