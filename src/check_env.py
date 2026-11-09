from __future__ import annotations

import platform
import sys

import torch


def pick_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    has_xpu = hasattr(torch, "xpu")
    print(f"XPU backend present: {has_xpu}")
    print(f"XPU available: {has_xpu and torch.xpu.is_available()}")
    if has_xpu and torch.xpu.is_available():
        print(f"XPU device count: {torch.xpu.device_count()}")
        print(f"XPU device 0: {torch.xpu.get_device_name(0)}")

    device = pick_device()
    tensor = torch.randn(1024, 1024, device=device)
    result = (tensor @ tensor.T).mean().detach().cpu().item()
    print(f"Selected device: {device}")
    print(f"Tensor smoke test: ok ({result:.6f})")


if __name__ == "__main__":
    main()
