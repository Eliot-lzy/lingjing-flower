from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.data import build_dataloaders
from src.engine import run_epoch, save_training_curves, seed_everything
from src.resnet import ResNet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "resnet18"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "resnet18_manual_best.pt"
HISTORY_PATH = OUTPUT_DIR / "history.csv"
CURVES_PATH = OUTPUT_DIR / "training_curves.png"
CLASS_MAP_PATH = PROJECT_ROOT / "configs" / "flower10_classes.json"

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train manual ResNet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    class_map = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    class_names = [item["class_name"] for item in class_map]

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"lr {learning_rate:.2e} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_name": "ResNet18FromScratch",
                    "model_state_dict": model.state_dict(),
                    "num_classes": 10,
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "image_size": 128,
                },
                MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1

        if (
            args.patience > 0
            and epochs_without_improvement >= args.patience
        ):
            print(f"Early stopping after epoch {epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)
    save_training_curves(history_df, CURVES_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy = run_epoch(
        model,
        test_loader,
        criterion,
        device,
    )

    elapsed_minutes = (time.perf_counter() - start_time) / 60
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Training time: {elapsed_minutes:.2f} minutes")
    print(f"Model: {MODEL_PATH}")
    print(f"Curves: {CURVES_PATH}")


if __name__ == "__main__":
    main()