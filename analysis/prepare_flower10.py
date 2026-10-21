from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets import Flowers102


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "configs"

MANIFEST_PATH = PROCESSED_DIR / "flower10_manifest.csv"
CLASS_MAP_PATH = CONFIG_DIR / "flower10_classes.json"

NUM_CLASSES = 10
SEED = 42


def collect_metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for split in ("train", "val", "test"):
        print(f"Loading Flowers102 {split} split...")
        dataset = Flowers102(
            root=RAW_ROOT,
            split=split,
            download=(split == "train"),
        )

        for image_path, source_label in zip(dataset._image_files, dataset._labels):
            relative_path = Path(image_path).resolve().relative_to(PROJECT_ROOT)
            rows.append(
                {
                    "image_path": relative_path.as_posix(),
                    "source_label": int(source_label),
                    "class_name": Flowers102.classes[int(source_label)],
                    "source_split": split,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    dataframe = collect_metadata()

    selected_ids = sorted(dataframe["source_label"].unique())[:NUM_CLASSES]
    dataframe = dataframe[dataframe["source_label"].isin(selected_ids)].copy()

    label_mapping = {source_id: index for index, source_id in enumerate(selected_ids)}
    dataframe["label"] = dataframe["source_label"].map(label_mapping).astype("int64")

    train_df, remaining_df = train_test_split(
        dataframe,
        test_size=0.30,
        stratify=dataframe["label"],
        random_state=SEED,
    )
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        stratify=remaining_df["label"],
        random_state=SEED,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
    manifest = manifest.sort_values(["split", "label", "image_path"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    class_map = (
        manifest[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
        .to_dict(orient="records")
    )
    CLASS_MAP_PATH.write_text(
        json.dumps(class_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = pd.crosstab(manifest["class_name"], manifest["split"])
    print("\nSplit count by class:")
    print(summary.to_string())
    print(f"\nTotal images: {len(manifest)}")
    print(f"Manifest    : {MANIFEST_PATH}")
    print(f"Class map   : {CLASS_MAP_PATH}")


if __name__ == "__main__":
    main()