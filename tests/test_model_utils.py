import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from model_utils import install_checkpoint, load_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_labels_and_weights(self):
        model = nn.Module()
        model.fc = nn.Linear(3, 2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pth"
            save_checkpoint(
                checkpoint_path,
                model,
                ["class_a", "class_b"],
                epoch=2,
                validation_accuracy=0.75,
                dataset_revision="test-revision",
            )
            checkpoint = load_checkpoint(checkpoint_path)

        self.assertEqual(checkpoint["class_names"], ["class_a", "class_b"])
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["validation_accuracy"], 0.75)
        self.assertTrue(torch.equal(checkpoint["model_state_dict"]["fc.weight"], model.fc.weight))

    def test_checkpoint_rejects_class_count_mismatch(self):
        model = nn.Module()
        model.fc = nn.Linear(3, 1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pth"
            with self.assertRaisesRegex(ValueError, "output count"):
                save_checkpoint(
                    checkpoint_path,
                    model,
                    ["class_a", "class_b"],
                    epoch=1,
                    validation_accuracy=0.5,
                    dataset_revision="test-revision",
                )

    def test_checkpoint_rejects_unsupported_format(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pth"
            torch.save({"format_version": 999}, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "Unsupported checkpoint format"):
                load_checkpoint(checkpoint_path)

    def test_invalid_import_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pth"
            checkpoint_path.write_bytes(b"existing checkpoint")
            with self.assertRaises(ValueError):
                install_checkpoint(BytesIO(b"invalid checkpoint"), checkpoint_path)
            self.assertEqual(checkpoint_path.read_bytes(), b"existing checkpoint")
            self.assertEqual(list(checkpoint_path.parent.glob("*.tmp")), [])

    def test_import_checks_weights_before_installation(self):
        model = nn.Module()
        model.fc = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source.pth"
            destination = Path(temporary_directory) / "installed.pth"
            save_checkpoint(source_path, model, ["a", "b"], epoch=1, validation_accuracy=0.8, dataset_revision="test")
            with source_path.open("rb") as source, patch("model_utils.build_model", return_value=model):
                install_checkpoint(source, destination)
            self.assertEqual(load_checkpoint(destination)["class_names"], ["a", "b"])

    def test_import_rejects_incompatible_weights(self):
        model = nn.Module()
        model.fc = nn.Linear(3, 2)
        incompatible_model = nn.Module()
        incompatible_model.fc = nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source.pth"
            destination = Path(temporary_directory) / "installed.pth"
            save_checkpoint(source_path, model, ["a", "b"], epoch=1, validation_accuracy=0.8, dataset_revision="test")
            with source_path.open("rb") as source, patch("model_utils.build_model", return_value=incompatible_model):
                with self.assertRaisesRegex(ValueError, "weights do not match"):
                    install_checkpoint(source, destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()