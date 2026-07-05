"""
models.py
Фабрика 5 архитектур классификации — пять разных архитектурных семейств
(не вариации одной модели), как того требует задание.
"""
import torch.nn as nn
from torchvision import models

ARCH_NAMES = ["resnet18", "mobilenet_v3_small", "efficientnet_b0", "densenet121", "vit_b_16"]


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    weights_kw = {"weights": "DEFAULT"} if pretrained else {"weights": None}

    if arch == "resnet18":
        m = models.resnet18(**weights_kw)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(**weights_kw)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m

    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(**weights_kw)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m

    if arch == "densenet121":
        m = models.densenet121(**weights_kw)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    if arch == "vit_b_16":
        # ViT тяжёлый (224x224, патчи 16x16); при нехватке ресурсов
        # замените на timm.create_model("vit_tiny_patch16_224", pretrained=True)
        m = models.vit_b_16(**weights_kw)
        in_features = m.heads.head.in_features
        m.heads.head = nn.Linear(in_features, num_classes)
        return m

    raise ValueError(f"Неизвестная архитектура: {arch}. Доступны: {ARCH_NAMES}")


def model_size_mb(model: nn.Module) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / (1024 ** 2)  # float32, МБ
