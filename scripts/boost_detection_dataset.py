from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import yaml

from common import PROJECT_ROOT


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    return [line.strip() for line in text.splitlines() if line.strip()] if text else []


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_yaml(dataset_root: Path, config_path: Path) -> None:
    payload = {
        "path": dataset_root.relative_to(PROJECT_ROOT).as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "defect"},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def iter_split_images(dataset_root: Path, split: str) -> Iterable[Path]:
    image_dir = dataset_root / "images" / split
    for path in sorted(image_dir.glob("*")):
        if path.is_file():
            yield path


def replicate_train_split(source_root: Path, output_root: Path, positive_repeat: int) -> tuple[int, int, int]:
    train_image_dir = output_root / "images" / "train"
    train_label_dir = output_root / "labels" / "train"
    train_image_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    positive_images = 0
    replicated_images = 0

    for image_path in iter_split_images(source_root, "train"):
        label_path = source_root / "labels" / "train" / f"{image_path.stem}.txt"
        labels = load_lines(label_path)
        is_positive = bool(labels)

        copy_file(image_path, train_image_dir / image_path.name)
        copy_file(label_path, train_label_dir / label_path.name)
        image_count += 1

        if is_positive:
            positive_images += 1
            for repeat_index in range(1, positive_repeat + 1):
                suffix = f"_rep{repeat_index:02d}"
                target_image = train_image_dir / f"{image_path.stem}{suffix}{image_path.suffix}"
                target_label = train_label_dir / f"{image_path.stem}{suffix}.txt"
                copy_file(image_path, target_image)
                target_label.write_text("\n".join(labels), encoding="utf-8")
                replicated_images += 1

    return image_count, positive_images, replicated_images


def copy_holdout_split(source_root: Path, output_root: Path, split: str) -> tuple[int, int]:
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    positive_images = 0
    for image_path in iter_split_images(source_root, split):
        label_path = source_root / "labels" / split / f"{image_path.stem}.txt"
        labels = load_lines(label_path)
        copy_file(image_path, image_dir / image_path.name)
        copy_file(label_path, label_dir / label_path.name)
        image_count += 1
        if labels:
            positive_images += 1
    return image_count, positive_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a boosted YOLO detection dataset by repeating positive train images.")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "datasets" / "dataset_1_defect", help="Source YOLO detection dataset root.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "datasets" / "dataset_1_defect_boosted", help="Output boosted dataset root.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dataset_1_defect_boosted.yaml", help="Output dataset yaml path.")
    parser.add_argument("--positive-repeat", type=int, default=4, help="How many extra copies to make for each positive training image.")
    args = parser.parse_args()

    if args.positive_repeat < 0:
        raise SystemExit("positive_repeat must be >= 0")

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config.resolve()

    if not source_root.exists():
        raise SystemExit(f"Source dataset root not found: {source_root}")

    if output_root.exists():
        shutil.rmtree(output_root)

    original_train_images, original_positive_images, replicated_images = replicate_train_split(
        source_root, output_root, args.positive_repeat
    )
    val_images, val_positive = copy_holdout_split(source_root, output_root, "val")
    test_images, test_positive = copy_holdout_split(source_root, output_root, "test")
    write_yaml(output_root, config_path)

    boosted_train_images = original_train_images + replicated_images
    boosted_positive_images = original_positive_images * (args.positive_repeat + 1)
    boosted_positive_ratio = round(boosted_positive_images / boosted_train_images, 6) if boosted_train_images else 0.0

    print(f"Boosted dataset written to: {output_root}")
    print(f"Dataset yaml written to: {config_path}")
    print(
        f"train: original_images={original_train_images}, original_positive={original_positive_images}, "
        f"replicated_images={replicated_images}, boosted_images={boosted_train_images}, "
        f"boosted_positive_images={boosted_positive_images}, boosted_positive_ratio={boosted_positive_ratio}"
    )
    print(f"val: images={val_images}, positive_images={val_positive}")
    print(f"test: images={test_images}, positive_images={test_positive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
