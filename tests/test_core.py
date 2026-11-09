from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dl_starter.checkpoint import load_checkpoint, save_checkpoint
from dl_starter.config import load_config
from dl_starter.data import SyntheticImageDataset
from dl_starter.models import build_model


class CoreTests(unittest.TestCase):
    def test_smoke_config_and_model_shape(self) -> None:
        config = load_config("configs/smoke.json")
        model = build_model(config.model, config.data.num_classes)
        output = model(torch.randn(2, 3, config.data.image_size, config.data.image_size))
        self.assertEqual(output.shape, (2, config.data.num_classes))

    def test_synthetic_dataset_is_deterministic(self) -> None:
        dataset = SyntheticImageDataset(10, 32, 10, seed=7)
        first_image, first_label = dataset[3]
        second_image, second_label = dataset[3]
        self.assertTrue(torch.equal(first_image, second_image))
        self.assertEqual(first_label, second_label)

    def test_checkpoint_round_trip(self) -> None:
        config = load_config("configs/smoke.json")
        source = build_model(config.model, config.data.num_classes)
        target = build_model(config.model, config.data.num_classes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint({"model": source.state_dict(), "epoch": 1}, path)
            checkpoint = load_checkpoint(path, target)
        self.assertEqual(checkpoint["epoch"], 1)
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(source_parameter, target_parameter))


if __name__ == "__main__":
    unittest.main()

