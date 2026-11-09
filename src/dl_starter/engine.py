from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    accuracy: float


def _move_batch(
    batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = batch
    return images.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    grad_clip: float,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_amp = amp and device.type in {"cuda", "xpu"}

    for batch in loader:
        images, targets = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.detach().float().item() * batch_size
        total_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return EpochMetrics(total_loss / total_samples, total_correct / total_samples)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_amp = amp and device.type in {"cuda", "xpu"}

    for batch in loader:
        images, targets = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, targets)
        batch_size = targets.size(0)
        total_loss += loss.detach().float().item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return EpochMetrics(total_loss / total_samples, total_correct / total_samples)

