from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torchvision.models import resnet18
from tqdm import tqdm

from src.data import build_dataloaders
from src.resnet import ResNet18
from src.simple_cnn import SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
CLASS_MAP_PATH = PROJECT_ROOT / "configs" / "flower10_classes.json"


def build_pretrained_architecture() -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


MODEL_SPECS = [
    {
        "key": "simple_cnn",
        "display_name": "SimpleCNN",
        "path": PROJECT_ROOT / "outputs" / "models" / "simple_cnn_best.pt",
        "builder": lambda: SimpleCNN(num_classes=10),
    },
    {
        "key": "manual_resnet18",
        "display_name": "Manual ResNet18",
        "path": PROJECT_ROOT / "outputs" / "models" / "resnet18_manual_best.pt",
        "builder": lambda: ResNet18(num_classes=10),
    },
    {
        "key": "pretrained_resnet18",
        "display_name": "Pretrained ResNet18",
        "path": PROJECT_ROOT / "outputs" / "models" / "resnet18_transfer_best.pt",
        "builder": build_pretrained_architecture,
    },
]


def predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    confidences: list[float] = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="predict", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)

            true_labels.extend(labels.tolist())
            predicted_labels.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())

    return true_labels, predicted_labels, confidences


def save_confusion_matrix(
    matrix: np.ndarray,
    normalized: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.55 else "#1B1B1E",
            )

    positions = np.arange(len(class_names))
    axis.set_xticks(positions)
    axis.set_yticks(positions)
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title, fontweight="bold")

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized value")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_map = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    class_names = [item["class_name"] for item in class_map]

    _, _, test_loader = build_dataloaders(batch_size=32, num_workers=0)
    test_paths = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in test_loader.dataset.image_paths
    ]

    comparison_rows: list[dict[str, object]] = []

    for specification in MODEL_SPECS:
        model_path = Path(specification["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        display_name = str(specification["display_name"])
        key = str(specification["key"])
        print(f"\nEvaluating {display_name}...")

        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
        model = specification["builder"]().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        true_labels, predicted_labels, confidences = predict(
            model,
            test_loader,
            device,
        )

        accuracy = accuracy_score(true_labels, predicted_labels)
        macro_f1 = f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )

        report = classification_report(
            true_labels,
            predicted_labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        (OUTPUT_DIR / f"classification_report_{key}.txt").write_text(
            report,
            encoding="utf-8",
        )

        matrix = confusion_matrix(true_labels, predicted_labels)
        normalized = confusion_matrix(
            true_labels,
            predicted_labels,
            normalize="true",
        )

        pd.DataFrame(
            matrix,
            index=class_names,
            columns=class_names,
        ).to_csv(OUTPUT_DIR / f"confusion_matrix_{key}.csv")

        save_confusion_matrix(
            matrix,
            normalized,
            class_names,
            f"{display_name} - Normalized confusion matrix",
            OUTPUT_DIR / f"confusion_matrix_{key}.png",
        )

        pd.DataFrame(
            {
                "image_path": test_paths,
                "true_label": true_labels,
                "true_class": [class_names[index] for index in true_labels],
                "predicted_label": predicted_labels,
                "predicted_class": [
                    class_names[index] for index in predicted_labels
                ],
                "confidence": confidences,
                "correct": np.array(true_labels) == np.array(predicted_labels),
            }
        ).to_csv(OUTPUT_DIR / f"predictions_{key}.csv", index=False)

        comparison_rows.append(
            {
                "model": display_name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "best_epoch": checkpoint.get("epoch"),
                "val_accuracy": checkpoint.get("val_accuracy"),
            }
        )

        print(report)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(comparison))
    width = 0.36

    axis.bar(
        positions - width / 2,
        comparison["accuracy"],
        width,
        label="Accuracy",
        color="#2A6F97",
    )
    axis.bar(
        positions + width / 2,
        comparison["macro_f1"],
        width,
        label="Macro F1",
        color="#E09F3E",
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(comparison["model"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Model comparison on the held-out test set", fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)

    for position, value in zip(positions - width / 2, comparison["accuracy"]):
        axis.text(position, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)

    for position, value in zip(positions + width / 2, comparison["macro_f1"]):
        axis.text(position, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "model_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nModel comparison:")
    print(comparison.round(4).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()