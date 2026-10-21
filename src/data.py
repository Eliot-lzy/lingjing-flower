from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "flower10_manifest.csv"

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.65, 1.0),
            ratio=(0.75, 1.3333),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(146),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class FlowerDataset(Dataset):
    def __init__(self, split: str, transform: transforms.Compose) -> None:
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

        manifest = pd.read_csv(MANIFEST_PATH)
        subset = manifest[manifest["split"] == split].reset_index(drop=True)

        if subset.empty:
            raise ValueError(f"No images found for split: {split}")

        self.image_paths = [PROJECT_ROOT / path for path in subset["image_path"]]
        self.labels = subset["label"].astype("int64").tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        return tensor, self.labels[index]


def build_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = FlowerDataset("train", TRAIN_TRANSFORM)
    val_dataset = FlowerDataset("val", EVAL_TRANSFORM)
    test_dataset = FlowerDataset("test", EVAL_TRANSFORM)

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    return train_loader, val_loader, test_loader