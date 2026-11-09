from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataConfig:
    dataset: str = "synthetic"
    root: str = "data"
    image_size: int = 32
    num_classes: int = 10
    batch_size: int = 64
    num_workers: int = 0
    train_size: int = 1024
    val_size: int = 256


@dataclass(slots=True)
class ModelConfig:
    name: str = "tiny_resnet"
    width: int = 32
    dropout: float = 0.1


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    amp: bool = True
    grad_clip: float = 1.0
    output_root: str = "outputs/runs"


@dataclass(slots=True)
class ExperimentConfig:
    run_name: str = "experiment"
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    config = ExperimentConfig(
        run_name=raw.get("run_name", "experiment"),
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.data.dataset not in {"synthetic", "cifar10"}:
        raise ValueError(f"Unsupported dataset: {config.data.dataset}")
    if config.model.name not in {"tiny_resnet", "resnet18"}:
        raise ValueError(f"Unsupported model: {config.model.name}")
    if config.data.num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if min(config.data.batch_size, config.train.epochs) < 1:
        raise ValueError("batch_size and epochs must be positive")

