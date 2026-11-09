from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import DataConfig


class SyntheticImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Deterministic images whose label is encoded by color and stripe position."""

    def __init__(self, size: int, image_size: int, num_classes: int, seed: int) -> None:
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = index % self.num_classes
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.randn(3, self.image_size, self.image_size, generator=generator) * 0.08
        channel = label % 3
        stripe = (label * max(1, self.image_size // self.num_classes)) % self.image_size
        end = min(stripe + max(2, self.image_size // 8), self.image_size)
        image[channel, :, stripe:end] += 0.9
        image[(channel + 1) % 3] += label / max(1, self.num_classes - 1) * 0.2
        return image.clamp(-1.0, 1.0), label


@dataclass(slots=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    class_names: list[str]


def build_dataloaders(config: DataConfig, seed: int, device: torch.device) -> DataLoaders:
    if config.dataset == "synthetic":
        train_dataset = SyntheticImageDataset(
            config.train_size, config.image_size, config.num_classes, seed
        )
        val_dataset = SyntheticImageDataset(
            config.val_size, config.image_size, config.num_classes, seed + 1_000_000
        )
        class_names = [f"class_{index}" for index in range(config.num_classes)]
    else:
        normalize = transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        val_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_dataset = datasets.CIFAR10(
            root=config.root, train=True, download=True, transform=train_transform
        )
        val_dataset = datasets.CIFAR10(
            root=config.root, train=False, download=True, transform=val_transform
        )
        class_names = list(train_dataset.classes)

    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return DataLoaders(train=train_loader, val=val_loader, class_names=class_names)

