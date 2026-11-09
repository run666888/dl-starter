from __future__ import annotations

import random

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU was requested but is not available")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def accelerator_name(device: torch.device) -> str:
    if device.type == "xpu":
        return torch.xpu.get_device_name(device.index or 0)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device.index or 0)
    return "CPU"


def make_grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    use_scaler = enabled and device.type in {"cuda", "xpu"}
    return torch.amp.GradScaler(device=device.type, enabled=use_scaler)

