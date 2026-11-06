import math
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
import torch.nn as nn
from torchvision import models, transforms


ARCHITECTURE = "resnet50"
CHECKPOINT_FORMAT_VERSION = 1
IMAGE_SIZE = 224
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    if num_classes < 2:
        raise ValueError("A classifier must have at least two classes.")
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_transform(training: bool = False):
    operations = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)])
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )
    return transforms.Compose(operations)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: Sequence[str],
    *,
    epoch: int,
    validation_accuracy: float,
    dataset_revision: str,
) -> None:
    names = list(class_names)
    if not all(isinstance(name, str) and name for name in names) or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Checkpoint class names must be non-empty and unique.")
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError("Checkpoint epoch must be a positive integer.")
    if not math.isfinite(validation_accuracy) or not 0 <= validation_accuracy <= 1:
        raise ValueError("Validation accuracy must be between zero and one.")
    if not dataset_revision:
        raise ValueError("Dataset revision must be provided.")

    model_state_dict = model.state_dict()
    classifier_weights = model_state_dict.get("fc.weight")
    if not isinstance(classifier_weights, torch.Tensor) or classifier_weights.ndim != 2 or classifier_weights.shape[0] != len(names):
        raise ValueError("The classifier output count must match the checkpoint class names.")

    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "class_names": names,
        "model_state_dict": model_state_dict,
        "epoch": epoch,
        "validation_accuracy": float(validation_accuracy),
        "dataset_revision": dataset_revision,
        "preprocessing": {
            "image_size": IMAGE_SIZE,
            "mean": list(NORMALIZATION_MEAN),
            "std": list(NORMALIZATION_STD),
        },
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="wb", dir=destination.parent, suffix=".tmp", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(checkpoint, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise
    except Exception as error:
        raise ValueError(f"Could not read checkpoint: {error}") from error

    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format. Retrain the model with this project version.")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Checkpoint architecture is not supported by this app.")

    class_names = checkpoint.get("class_names")
    model_state_dict = checkpoint.get("model_state_dict")
    preprocessing = checkpoint.get("preprocessing")
    epoch = checkpoint.get("epoch")
    validation_accuracy = checkpoint.get("validation_accuracy")
    dataset_revision = checkpoint.get("dataset_revision")
    if (
        not isinstance(class_names, list)
        or len(class_names) < 2
        or not all(isinstance(name, str) and name for name in class_names)
        or len(set(class_names)) != len(class_names)
        or not isinstance(model_state_dict, dict)
        or not isinstance(epoch, int)
        or epoch < 1
        or not isinstance(validation_accuracy, (int, float))
        or not math.isfinite(validation_accuracy)
        or not 0 <= validation_accuracy <= 1
        or not isinstance(dataset_revision, str)
        or not dataset_revision
        or preprocessing
        != {
            "image_size": IMAGE_SIZE,
            "mean": list(NORMALIZATION_MEAN),
            "std": list(NORMALIZATION_STD),
        }
    ):
        raise ValueError("Checkpoint metadata is incomplete or inconsistent.")

    classifier_weights = model_state_dict.get("fc.weight")
    if not isinstance(classifier_weights, torch.Tensor) or classifier_weights.ndim != 2 or classifier_weights.shape[0] != len(class_names):
        raise ValueError("Checkpoint classifier output does not match its class names.")
    return checkpoint


def load_trained_model(path: Path):
    checkpoint = load_checkpoint(path)
    model = build_model(len(checkpoint["class_names"]))
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, TypeError, AttributeError) as error:
        raise ValueError(f"Checkpoint weights do not match the model: {error}") from error
    model.eval()
    return model, checkpoint


def install_checkpoint(source, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="wb", dir=destination.parent, suffix=".tmp", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            shutil.copyfileobj(source, temporary_file)
        load_trained_model(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)