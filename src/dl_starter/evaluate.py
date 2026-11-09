from __future__ import annotations

import argparse

import torch
from torch import nn

from .checkpoint import load_checkpoint
from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from .data import build_dataloaders
from .device import accelerator_name, resolve_device
from .engine import evaluate
from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved image classifier")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_config = raw["config"]
    config = ExperimentConfig(
        run_name=raw_config["run_name"],
        seed=raw_config["seed"],
        device=args.device,
        data=DataConfig(**raw_config["data"]),
        model=ModelConfig(**raw_config["model"]),
        train=TrainConfig(**raw_config["train"]),
    )
    loaders = build_dataloaders(config.data, config.seed, device)
    model = build_model(config.model, config.data.num_classes).to(device)
    load_checkpoint(args.checkpoint, model)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)
    metrics = evaluate(model, loaders.val, loss_fn, device, config.train.amp)
    print(f"Device: {device} ({accelerator_name(device)})")
    print(f"Validation loss: {metrics.loss:.4f}")
    print(f"Validation accuracy: {metrics.accuracy:.3f}")


if __name__ == "__main__":
    main()
