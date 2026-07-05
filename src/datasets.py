"""
datasets.py
DataLoader-фабрика с аугментациями. Отдельные, более слабые аугментации для
валидации/теста (только resize+normalize), и более сильные для train —
в т.ч. имитация плохих условий съёмки знаков (размытие, низкая освещённость,
шум), что прямо требуется заданием ("мелкие, размытые и плохо освещённые знаки").
"""
from pathlib import Path

import torch
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),  # имитация частичной засветки/загораживания
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dataloaders(data_root: str, img_size: int = 224, batch_size: int = 32, num_workers: int = 4):
    data_root = Path(data_root)
    train_ds = datasets.ImageFolder(data_root / "train", transform=build_transforms(img_size, train=True))
    val_ds = datasets.ImageFolder(data_root / "val", transform=build_transforms(img_size, train=False))
    test_ds = datasets.ImageFolder(data_root / "test", transform=build_transforms(img_size, train=False))

    assert train_ds.classes == val_ds.classes == test_ds.classes, \
        "Несовпадение классов между train/val/test — проверьте подготовку данных"

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes
