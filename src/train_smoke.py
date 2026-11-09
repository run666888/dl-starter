from __future__ import annotations

import torch
from torch import nn

from check_env import pick_device


def make_batch(samples: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(samples, 2, generator=generator)
    y = ((x[:, 0] * x[:, 1]) > 0).long()
    return x, y


def main() -> None:
    device = pick_device()
    x, y = make_batch()
    x = x.to(device)
    y = y.to(device)

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(1, 201):
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            accuracy = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"step={step:03d} loss={loss.item():.4f} acc={accuracy:.3f} device={device}")

    print("Training smoke test: ok")


if __name__ == "__main__":
    main()
