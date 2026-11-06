import argparse
import random
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model_utils import build_model, create_transform, load_checkpoint, save_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ID = "mohanty/PlantVillage"
DATASET_REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
NUM_CLASSES = 38


class PlantVillageDataset(Dataset):
    def __init__(self, samples, class_to_index, transform):
        self.samples = samples
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, class_name = self.samples[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        return self.transform(image), self.class_to_index[class_name]


def safe_extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            target_path = (destination / Path(*member_path.parts)).resolve()
            if member_path.is_absolute() or not target_path.is_relative_to(destination_root):
                raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
        archive.extractall(destination_root)


def download_split_file(data_dir: Path, filename: str, allow_download: bool) -> Path:
    local_path = data_dir / "splits" / filename
    if local_path.is_file():
        return local_path
    if not allow_download:
        raise FileNotFoundError(f"Missing {local_path}. Rerun training with --download-data.")

    cached_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=f"splits/{filename}",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached_path, local_path)
    return local_path


def prepare_dataset(data_dir: Path, allow_download: bool):
    image_root = data_dir / "raw" / "color"
    if not image_root.is_dir():
        if not allow_download:
            raise FileNotFoundError(
                f"Dataset images are missing from {image_root}. Rerun training with --download-data."
            )
        archive_path = hf_hub_download(
            repo_id=DATASET_ID,
            filename="data.zip",
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
        safe_extract_archive(Path(archive_path), data_dir)
        if not image_root.is_dir():
            raise FileNotFoundError(f"The archive did not contain the expected directory: {image_root}")

    train_split = download_split_file(data_dir, "color_train.txt", allow_download)
    test_split = download_split_file(data_dir, "color_test.txt", allow_download)
    return image_root, train_split, test_split


def read_split(image_root: Path, split_file: Path):
    data_root = image_root.parents[1].resolve()
    samples = []

    with split_file.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            raw_path = line.strip()
            if not raw_path:
                continue
            relative_path = PurePosixPath(raw_path)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or len(relative_path.parts) < 4
                or relative_path.parts[:2] != ("raw", "color")
            ):
                raise ValueError(f"Invalid dataset path on line {line_number}: {raw_path}")

            class_name = relative_path.parts[2]
            image_path = data_root.joinpath(*relative_path.parts)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed on line {line_number} is missing: {image_path}")
            samples.append((image_path, class_name))

    if not samples:
        raise ValueError(f"No image paths found in split file: {split_file}")
    return samples


def run_training_epoch(model, data_loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        if total_samples // data_loader.batch_size % 100 == 0:
            print(f"Training progress: {total_samples}/{len(data_loader.dataset)} images.", flush=True)

    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def evaluate(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        outputs = model(images)
        loss = loss_function(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet-50 on the official PlantVillage split.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "plantvillage")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "plant_disease_resnet50.pth")
    parser.add_argument("--download-data", action="store_true", help="Download the 2.18 GB image archive and official split files.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive; workers cannot be negative.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = args.data_dir.resolve()
    print(f"Preparing PlantVillage data in {data_dir}. Downloads are reused when available.", flush=True)
    image_root, train_split, test_split = prepare_dataset(data_dir, args.download_data)
    train_samples = read_split(image_root, train_split)
    test_samples = read_split(image_root, test_split)

    class_names = sorted({class_name for _, class_name in train_samples})
    test_class_names = {class_name for _, class_name in test_samples}
    if len(class_names) != NUM_CLASSES or test_class_names != set(class_names):
        raise ValueError(
            f"Expected the same {NUM_CLASSES} classes in both splits; "
            f"found {len(class_names)} training classes and {len(test_class_names)} test classes."
        )

    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    train_dataset = PlantVillageDataset(train_samples, class_to_index, create_transform(training=True))
    test_dataset = PlantVillageDataset(test_samples, class_to_index, create_transform())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = build_model(len(class_names), pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    checkpoint_path = args.checkpoint.resolve()

    if checkpoint_path.is_file():
        try:
            previous_checkpoint = load_checkpoint(checkpoint_path)
        except ValueError:
            print("Existing checkpoint is incompatible; it will be replaced after a successful validation pass.")
        else:
            if (
                previous_checkpoint["dataset_revision"] == DATASET_REVISION
                and previous_checkpoint["class_names"] == class_names
            ):
                best_accuracy = previous_checkpoint["validation_accuracy"]
                print(f"Keeping existing checkpoint unless validation accuracy exceeds {best_accuracy:.2%}.")

    print(f"Training on {len(train_dataset)} images; validating on {len(test_dataset)} images using {device}.")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_training_epoch(model, train_loader, optimizer, loss_function, device)
        validation_loss, validation_accuracy = evaluate(model, test_loader, loss_function, device)
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train loss={train_loss:.4f}, train accuracy={train_accuracy:.2%}, "
            f"validation loss={validation_loss:.4f}, validation accuracy={validation_accuracy:.2%}"
        )

        if validation_accuracy > best_accuracy:
            save_checkpoint(
                checkpoint_path,
                model,
                class_names,
                epoch=epoch,
                validation_accuracy=validation_accuracy,
                dataset_revision=DATASET_REVISION,
            )
            best_accuracy = validation_accuracy
            print(f"Saved best checkpoint to {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
