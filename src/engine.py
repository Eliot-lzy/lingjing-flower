from __future__ import annotations

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
    grad_clip: float | None = None,
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

                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

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


def save_training_curves(
    history: pd.DataFrame,
    output_path: Path,
) -> None:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)