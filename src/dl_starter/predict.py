from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision import transforms

from .checkpoint import load_checkpoint
from .config import ModelConfig
from .device import accelerator_name, resolve_device
from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one image with a saved checkpoint")
    parser.add_argument("image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_config = checkpoint["config"]
    data_config = raw_config["data"]
    model = build_model(
        ModelConfig(**raw_config["model"]), data_config["num_classes"]
    ).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((data_config["image_size"], data_config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(image).softmax(dim=1).squeeze(0)

    class_names = checkpoint.get(
        "class_names", [str(index) for index in range(probabilities.numel())]
    )
    top_k = min(max(1, args.top_k), probabilities.numel())
    values, indices = probabilities.topk(top_k)
    print(f"Device: {device} ({accelerator_name(device)})")
    for rank, (value, index) in enumerate(zip(values.cpu(), indices.cpu()), start=1):
        print(f"{rank}. {class_names[index.item()]}: {value.item():.2%}")


if __name__ == "__main__":
    main()

