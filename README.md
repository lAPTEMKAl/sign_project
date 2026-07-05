# Распознавание дорожных знаков (RTSD) — сравнение 5 архитектур классификации

Проект реализует полный цикл практики: подготовка данных RTSD, обучение/дообучение
5 различных архитектур классификации, экспериментальное сравнение, демонстрационный
модуль и генерация отчёта с историей запусков.

## Состав архитектур (5 разных семейств, не вариации одной модели)

| № | Архитектура | Семейство | Источник весов |
|---|-------------|-----------|----------------|
| 1 | ResNet18 | CNN, residual | torchvision (ImageNet) |
| 2 | MobileNetV3-Small | CNN, легковесная | torchvision (ImageNet) |
| 3 | EfficientNet-B0 | CNN, compound scaling | torchvision (ImageNet) |
| 4 | DenseNet121 | CNN, dense connections | torchvision (ImageNet) |
| 5 | ViT-B/16 (или ViT-Tiny через timm) | Transformer | torchvision/timm (ImageNet) |

## Структура проекта

```
project/
  data/
    raw/            # сюда скачивается/распаковывается RTSD
    processed/       # train/val/test после подготовки (ImageFolder-формат)
  src/
    download_data.py # инструкция/скрипт получения RTSD
    prepare_data.py  # разбиение, проверка разметки, аугментации
    datasets.py       # PyTorch Dataset/DataLoader
    models.py          # фабрика моделей (5 архитектур)
    train.py            # обучение/дообучение одной модели
    evaluate.py          # метрики, confusion matrix, ошибки
    run_all.py            # обучает все 5 моделей и собирает сравнение
    inference.py           # инференс одного изображения + история в JSON/SQLite
    report_generator.py     # генерация отчёта PDF/Excel из истории запусков
  configs/
    models.yaml       # гиперпараметры по каждой модели
  demo/
    app.py            # Streamlit-демо: загрузка фото знака -> класс + уверенность + статистика
  report/
    template_outline.md  # план итогового отчёта по ГОСТ (см. отдельный .docx)
  runs/               # сюда пишутся результаты обучения, история запросов (JSON/SQLite)
  models/             # сохранённые веса .pt
  README.md
  requirements.txt
```

## Быстрый старт

```bash
pip install -r requirements.txt

# 1. Скачать и распаковать RTSD https://www.kaggle.com/datasets/watchman/rtsd-dataset (см. инструкцию в src/download_data.py)
python src/download_data.py --target data/raw

# 2. Подготовить train/val/test
python src/prepare_data.py --raw data/raw --out data/processed --val-split 0.15 --test-split 0.15

# 3. Обучить все 5 архитектур и получить сравнительную таблицу
python src/run_all.py --data data/processed --epochs 20 --img-size 224 --out runs/experiment_1

# 4. Оценить лучшую модель подробно (confusion matrix, ошибки)
python src/evaluate.py --model models/best_model.pt --data data/processed/test --out runs/experiment_1/eval_best

# 5. Запустить демо
streamlit run demo/app.py

# 6. Сгенерировать отчёт PDF/Excel по истории запусков
python src/report_generator.py --history runs/experiment_1/history.db --out report/
```
