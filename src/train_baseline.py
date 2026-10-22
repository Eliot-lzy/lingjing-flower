from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.data import build_dataloaders
from src.simple_cnn import SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "simple_cnn_best.pt"
HISTORY_PATH = OUTPUT_DIR / "history.csv"
CURVES_PATH = OUTPUT_DIR / "training_curves.png"
CLASS_MAP_PATH = PROJECT_ROOT / "configs" / "flower10_classes.json"

SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc="train" if is_training else "val", leave=False)

    with torch.set_grad_enabled(is_training):
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

            progress.set_postfix(
                loss=f"{running_loss / total:.4f}",
                accuracy=f"{correct / total:.4f}",
            )

    return running_loss / total, correct / total


def save_training_curves(history: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="validation")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["train_accuracy"], marker="o", label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], marker="o", label="validation")
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(CURVES_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimpleCNN baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    class_map = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    class_names = [item["class_name"] for item in class_map]

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "model_name": "SimpleCNN",
                    "model_state_dict": model.state_dict(),
                    "num_classes": 10,
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "image_size": 128,
                },
                MODEL_PATH,
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)
    save_training_curves(history_df)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy = run_epoch(
        model, test_loader, criterion, device
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Model: {MODEL_PATH}")
    print(f"Curves: {CURVES_PATH}")


if __name__ == "__main__":
    main()