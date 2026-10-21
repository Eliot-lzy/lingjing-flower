from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "flower10_image_stats.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis"

SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLORS = {
    "train": "#2A6F97",
    "val": "#61A5C2",
    "test": "#E09F3E",
}


def save_class_distribution(data: pd.DataFrame, class_names: list[str]) -> None:
    counts = pd.crosstab(data["class_name"], data["split"])
    counts = counts.reindex(index=class_names, columns=SPLIT_ORDER, fill_value=0)

    figure, axis = plt.subplots(figsize=(10, 6))
    left = pd.Series(0, index=counts.index, dtype=float)

    for split in SPLIT_ORDER:
        values = counts[split].astype(float)
        axis.barh(
            counts.index,
            values,
            left=left,
            color=SPLIT_COLORS[split],
            label=split,
            edgecolor="white",
            linewidth=0.6,
        )
        left += values

    axis.set_title("Class distribution by dataset split", fontweight="bold")
    axis.set_xlabel("Number of images")
    axis.set_ylabel("Flower class")
    axis.legend(title="Split", frameon=False)
    axis.grid(axis="x", alpha=0.25)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_image_dimensions(data: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    for split in SPLIT_ORDER:
        subset = data[data["split"] == split]
        axes[0].scatter(
            subset["width"],
            subset["height"],
            s=15,
            alpha=0.55,
            color=SPLIT_COLORS[split],
            label=split,
        )

    axes[0].set_title("Original image dimensions", fontweight="bold")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Height (pixels)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    megapixels = data["pixel_count"] / 1_000_000
    axes[1].hist(megapixels, bins=30, color="#5C4D7D", edgecolor="white")
    axes[1].set_title("Image size distribution", fontweight="bold")
    axes[1].set_xlabel("Megapixels")
    axes[1].set_ylabel("Number of images")
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "image_dimensions.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_brightness_boxplot(data: pd.DataFrame, class_names: list[str]) -> None:
    grouped = [
        data.loc[data["class_name"] == name, "brightness"].to_numpy()
        for name in class_names
    ]

    figure, axis = plt.subplots(figsize=(11, 6))
    boxplot = axis.boxplot(
        grouped,
        orientation="horizontal",
        tick_labels=class_names,
        patch_artist=True,
        medianprops={"color": "#1B1B1E", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.45},
    )

    palette = plt.get_cmap("tab10").colors
    for patch, color in zip(boxplot["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    axis.set_title("Brightness distribution by flower class", fontweight="bold")
    axis.set_xlabel("Mean brightness after resizing to 128 x 128")
    axis.set_ylabel("Flower class")
    axis.grid(axis="x", alpha=0.25)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "brightness_by_class.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_sample_grid(data: pd.DataFrame, class_table: pd.DataFrame) -> None:
    samples = (
        data[data["split"] == "train"]
        .sort_values("label")
        .drop_duplicates("label")
        .set_index("label")
        .loc[class_table["label"]]
        .reset_index()
    )

    if len(samples) != 10:
        raise ValueError(f"Expected 10 samples, received {len(samples)}")

    figure, axes = plt.subplots(2, 5, figsize=(15, 6.5))

    for axis, row in zip(axes.flat, samples.itertuples(index=False)):
        with Image.open(PROJECT_ROOT / str(row.image_path)) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(str(row.class_name), fontsize=10)
        axis.axis("off")

    figure.suptitle("One training image from each class", fontweight="bold")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "class_samples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Run extract_image_stats.py first: {STATS_PATH}")

    data = pd.read_csv(STATS_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_table = (
        data[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
        .reset_index(drop=True)
    )
    class_names = class_table["class_name"].tolist()

    summary = (
        data.groupby(["split", "class_name"], as_index=False)
        .agg(
            count=("image_path", "size"),
            mean_width=("width", "mean"),
            mean_height=("height", "mean"),
            mean_brightness=("brightness", "mean"),
            mean_contrast=("contrast", "mean"),
        )
        .round(4)
    )
    summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)

    save_class_distribution(data, class_names)
    save_image_dimensions(data)
    save_brightness_boxplot(data, class_names)
    save_sample_grid(data, class_table)

    print(f"Saved charts and summary to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()