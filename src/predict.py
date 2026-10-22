from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.data import EVAL_TRANSFORM
from src.model_factory import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "models" / "resnet18_transfer_best.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one flower image")
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def save_result_figure(
    image: Image.Image,
    class_names: list[str],
    top_indices: list[int],
    top_probabilities: list[float],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title(
        f"{class_names[top_indices[0]]}: {top_probabilities[0]:.2%}",
        fontsize=14,
        fontweight="bold",
    )

    labels = [class_names[index] for index in top_indices]
    colors = ["#2A6F97"] + ["#A9D6E5"] * (len(labels) - 1)

    y_positions = list(range(len(labels)))
    axes[1].barh(
        y_positions,
        top_probabilities,
        color=colors,
        edgecolor="white",
    )
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Top predictions", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.25)

    for position, probability in zip(y_positions, top_probabilities):
        axes[1].text(
            min(probability + 0.02, 0.88),
            position,
            f"{probability:.2%}",
            va="center",
            fontsize=10,
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)

    class_names = list(checkpoint["class_names"])
    top_k = min(args.top_k, len(class_names))

    if top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    with Image.open(args.image) as image:
        rgb_image = image.convert("RGB")
        input_tensor = EVAL_TRANSFORM(rgb_image).unsqueeze(0).to(device)
        preview_image = rgb_image.copy()

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    values, indices = torch.topk(probabilities, k=top_k)
    top_probabilities = values.cpu().tolist()
    top_indices = indices.cpu().tolist()

    safe_stem = args.image.stem.replace(" ", "_")
    figure_path = args.output_dir / f"{safe_stem}_prediction.png"
    json_path = args.output_dir / f"{safe_stem}_prediction.json"

    save_result_figure(
        image=preview_image,
        class_names=class_names,
        top_indices=top_indices,
        top_probabilities=top_probabilities,
        output_path=figure_path,
    )

    result = {
        "image": str(args.image),
        "checkpoint": str(args.checkpoint),
        "model_name": checkpoint["model_name"],
        "top_predictions": [
            {
                "rank": rank,
                "class_name": class_names[index],
                "probability": probability,
            }
            for rank, (index, probability) in enumerate(
                zip(top_indices, top_probabilities),
                start=1,
            )
        ],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Model: {checkpoint['model_name']}")
    print(f"Image: {args.image}")
    print("Top predictions:")

    for rank, (index, probability) in enumerate(
        zip(top_indices, top_probabilities),
        start=1,
    ):
        print(f"  {rank}. {class_names[index]:16s} {probability:.2%}")

    print(f"Result image: {figure_path}")
    print(f"Result json : {json_path}")


if __name__ == "__main__":
    main()