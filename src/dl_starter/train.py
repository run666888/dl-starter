from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
from .data import build_dataloaders
from .device import accelerator_name, make_grad_scaler, resolve_device, seed_everything
from .engine import evaluate, train_one_epoch
from .models import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--resume", help="Path to a last.pt checkpoint")
    parser.add_argument("--device", help="Override config device: auto, cpu, xpu, cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device

    seed_everything(config.seed)
    device = resolve_device(config.device)
    loaders = build_dataloaders(config.data, config.seed, device)
    model = build_model(config.model, config.data.num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.epochs
    )
    scaler = make_grad_scaler(device, config.train.amp)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.train.output_root) / f"{config.run_name}-{timestamp}"
    start_epoch = 1
    best_accuracy = 0.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint.get("best_accuracy", 0.0))
        run_dir = Path(args.resume).resolve().parent

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    print(f"Device: {device} ({accelerator_name(device)})")
    print(
        f"Dataset: {config.data.dataset} | "
        f"train={len(loaders.train.dataset)} val={len(loaders.val.dataset)}"
    )
    print(f"Model: {config.model.name} | trainable parameters={count_parameters(model):,}")
    print(f"Run directory: {run_dir.resolve()}")

    metrics_path = run_dir / "metrics.csv"
    write_header = not metrics_path.exists() or start_epoch == 1
    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ],
        )
        if write_header:
            writer.writeheader()

        for epoch in range(start_epoch, config.train.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                loaders.train,
                optimizer,
                loss_fn,
                scaler,
                device,
                config.train.amp,
                config.train.grad_clip,
            )
            val_metrics = evaluate(model, loaders.val, loss_fn, device, config.train.amp)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            writer.writerow(
                {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train_loss": train_metrics.loss,
                    "train_accuracy": train_metrics.accuracy,
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                }
            )
            metrics_file.flush()

            is_best = val_metrics.accuracy > best_accuracy
            best_accuracy = max(best_accuracy, val_metrics.accuracy)
            state = {
                "epoch": epoch,
                "best_accuracy": best_accuracy,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config.to_dict(),
                "class_names": loaders.class_names,
            }
            save_checkpoint(state, run_dir / "last.pt")
            if is_best:
                save_checkpoint(state, run_dir / "best.pt")

            print(
                f"Epoch {epoch:03d}/{config.train.epochs:03d} | "
                f"train loss={train_metrics.loss:.4f} acc={train_metrics.accuracy:.3f} | "
                f"val loss={val_metrics.loss:.4f} acc={val_metrics.accuracy:.3f}"
            )

    summary = {
        "best_accuracy": best_accuracy,
        "epochs_completed": config.train.epochs,
        "device": str(device),
        "accelerator": accelerator_name(device),
        "best_checkpoint": str((run_dir / "best.pt").resolve()),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Training complete | best validation accuracy={best_accuracy:.3f}")


if __name__ == "__main__":
    main()

