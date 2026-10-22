from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.data import build_dataloaders
from src.engine import run_epoch, save_training_curves, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "transfer"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "resnet18_transfer_best.pt"
HISTORY_PATH = OUTPUT_DIR / "history.csv"
CURVES_PATH = OUTPUT_DIR / "training_curves.png"
CLASS_MAP_PATH = PROJECT_ROOT / "configs" / "flower10_classes.json"

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pretrained ResNet18")
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def configure_trainable_parameters(
    model: nn.Module,
    fine_tune: bool,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = fine_tune

    if not fine_tune:
        for parameter in model.fc.parameters():
            parameter.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    args = parse_args()
    if args.head_epochs + args.finetune_epochs <= 0:
        raise ValueError("At least one training epoch is required.")

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

    print("Loading ImageNet pretrained ResNet18...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    class_map = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    class_names = [item["class_name"] for item in class_map]

    stages = [
        {
            "name": "head_warmup",
            "epochs": args.head_epochs,
            "learning_rate": args.head_lr,
            "fine_tune": False,
            "patience": 0,
        },
        {
            "name": "full_finetune",
            "epochs": args.finetune_epochs,
            "learning_rate": args.finetune_lr,
            "fine_tune": True,
            "patience": args.patience,
        },
    ]

    history: list[dict[str, float | int | str]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    global_epoch = 0
    start_time = time.perf_counter()

    for stage in stages:
        stage_epochs = int(stage["epochs"])
        if stage_epochs <= 0:
            continue

        stage_name = str(stage["name"])
        fine_tune = bool(stage["fine_tune"])
        configure_trainable_parameters(model, fine_tune=fine_tune)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(stage["learning_rate"]),
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=stage_epochs,
            eta_min=1e-6,
        )

        print(
            f"\nStage: {stage_name} | epochs: {stage_epochs} | "
            f"trainable parameters: {count_trainable_parameters(model):,}"
        )

        epochs_without_improvement = 0

        for _ in range(stage_epochs):
            global_epoch += 1
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
                    "epoch": global_epoch,
                    "stage": stage_name,
                    "learning_rate": learning_rate,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )

            print(
                f"Epoch {global_epoch:02d} | {stage_name} | "
                f"lr {learning_rate:.2e} | "
                f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
                f"val loss {val_loss:.4f} acc {val_accuracy:.4f}"
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = global_epoch
                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_name": "ResNet18TransferLearning",
                        "model_state_dict": model.state_dict(),
                        "num_classes": 10,
                        "class_names": class_names,
                        "epoch": global_epoch,
                        "stage": stage_name,
                        "val_accuracy": val_accuracy,
                        "image_size": 128,
                        "pretrained_weights": "ResNet18_Weights.DEFAULT",
                    },
                    MODEL_PATH,
                )
            else:
                epochs_without_improvement += 1

            stage_patience = int(stage["patience"])
            if (
                stage_patience > 0
                and epochs_without_improvement >= stage_patience
            ):
                print(f"Early stopping {stage_name} at epoch {global_epoch}.")
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