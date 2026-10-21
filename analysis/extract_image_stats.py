from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "flower10_manifest.csv"
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "flower10_image_stats.csv"


def analyze_image(image_path: Path) -> dict[str, float | int]:
    with Image.open(image_path) as image:
        width, height = image.size
        mode = image.mode
        thumbnail = image.convert("RGB").resize(
            (128, 128),
            Image.Resampling.BILINEAR,
        )

    pixels = np.asarray(thumbnail, dtype=np.float32) / 255.0
    red_mean, green_mean, blue_mean = pixels.mean(axis=(0, 1))

    return {
        "width": int(width),
        "height": int(height),
        "pixel_count": int(width * height),
        "aspect_ratio": float(width / height),
        "mode": mode,
        "file_size_kb": image_path.stat().st_size / 1024,
        "brightness": float(np.mean(pixels)),
        "contrast": float(np.std(pixels)),
        "red_mean": float(red_mean),
        "green_mean": float(green_mean),
        "blue_mean": float(blue_mean),
    }


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Run prepare_flower10.py first: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    records: list[dict[str, object]] = []

    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Analyzing images",
    ):
        absolute_path = PROJECT_ROOT / str(row.image_path)
        try:
            records.append(
                {
                    "image_path": str(row.image_path),
                    **analyze_image(absolute_path),
                }
            )
        except Exception as error:
            raise RuntimeError(f"Failed to read image: {absolute_path}") from error

    statistics = pd.DataFrame(records)
    result = manifest.merge(
        statistics,
        on="image_path",
        how="left",
        validate="one_to_one",
    )

    if result.isna().any().any():
        raise ValueError("Some images were not analyzed successfully.")

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(STATS_PATH, index=False)

    split_counts = result.groupby("split").size().reindex(["train", "val", "test"])
    class_counts = pd.crosstab(result["class_name"], result["split"])
    dimensions = result[["width", "height", "pixel_count"]].describe().round(2)

    print("\nSplit counts:")
    print(split_counts.to_string())
    print("\nClass counts:")
    print(class_counts.to_string())
    print("\nImage dimensions:")
    print(dimensions.to_string())
    print(f"\nSaved: {STATS_PATH}")


if __name__ == "__main__":
    main()