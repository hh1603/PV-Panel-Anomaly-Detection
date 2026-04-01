from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import yaml

from common import PROJECT_ROOT


@dataclass
class Sample:
    image_path: Path
    labels: list[str]
    has_defect: bool


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def corners_to_yolo_line(corners: list[dict], width: int, height: int) -> str | None:
    xs = [point["x"] for point in corners]
    ys = [point["y"] for point in corners]
    min_x = clip(min(xs), 0.0, float(width))
    max_x = clip(max(xs), 0.0, float(width))
    min_y = clip(min(ys), 0.0, float(height))
    max_y = clip(max(ys), 0.0, float(height))

    box_w = max_x - min_x
    box_h = max_y - min_y
    if box_w <= 1.0 or box_h <= 1.0:
        return None

    x_center = (min_x + max_x) / 2.0 / width
    y_center = (min_y + max_y) / 2.0 / height
    norm_w = box_w / width
    norm_h = box_h / height
    return f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def stratified_split(samples: list[Sample], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    positive = [sample for sample in samples if sample.has_defect]
    negative = [sample for sample in samples if not sample.has_defect]
    rng.shuffle(positive)
    rng.shuffle(negative)

    def assign(group: list[Sample]) -> tuple[list[Sample], list[Sample], list[Sample]]:
        train_count, val_count, _ = split_counts(len(group), train_ratio, val_ratio)
        train_items = group[:train_count]
        val_items = group[train_count:train_count + val_count]
        test_items = group[train_count + val_count:]
        return train_items, val_items, test_items

    pos_train, pos_val, pos_test = assign(positive)
    neg_train, neg_val, neg_test = assign(negative)

    splits = {
        "train": pos_train + neg_train,
        "val": pos_val + neg_val,
        "test": pos_test + neg_test,
    }
    for items in splits.values():
        rng.shuffle(items)
    return splits


def load_samples(source_root: Path) -> list[Sample]:
    image_root = source_root / "images"
    annotation_root = source_root / "annotations"
    samples: list[Sample] = []

    for image_path in sorted(image_root.glob("*.jpg")):
        annotation_path = annotation_root / f"{image_path.stem}.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation for image: {image_path}")

        with Image.open(image_path) as image:
            width, height = image.size

        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        labels: list[str] = []
        for instance in payload.get("instances", []):
            if not instance.get("defected_module"):
                continue
            corners = instance.get("corners", [])
            if len(corners) < 4:
                continue
            line = corners_to_yolo_line(corners, width, height)
            if line is not None:
                labels.append(line)

        samples.append(Sample(image_path=image_path, labels=labels, has_defect=bool(labels)))

    return samples


def write_dataset(splits: dict[str, list[Sample]], output_root: Path) -> None:
    for split, samples in splits.items():
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            target_image = image_dir / sample.image_path.name
            shutil.copy2(sample.image_path, target_image)
            target_label = label_dir / f"{sample.image_path.stem}.txt"
            target_label.write_text("\n".join(sample.labels), encoding="utf-8")


def write_dataset_yaml(output_root: Path, config_path: Path) -> None:
    relative_dataset_root = output_root.relative_to(PROJECT_ROOT)
    payload = {
        "path": relative_dataset_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "defect"},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert dataset_1 custom JSON annotations to YOLO detect format.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "dataset_1", help="Source dataset_1 root.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "dataset_1_defect", help="Output YOLO dataset root.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dataset_1_defect.yaml", help="Output dataset yaml path.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split shuffling.")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("train_ratio and val_ratio must satisfy: train_ratio > 0, val_ratio >= 0, train_ratio + val_ratio < 1")

    source_root = args.source.resolve()
    output_root = args.output.resolve()
    config_path = args.config.resolve()

    if output_root.exists():
        shutil.rmtree(output_root)

    samples = load_samples(source_root)
    if not samples:
        raise SystemExit(f"No samples found under: {source_root}")

    splits = stratified_split(samples, args.train_ratio, args.val_ratio, args.seed)
    write_dataset(splits, output_root)
    write_dataset_yaml(output_root, config_path)

    total_boxes = sum(len(sample.labels) for sample in samples)
    total_positive = sum(1 for sample in samples if sample.has_defect)
    print(f"Converted {len(samples)} images with {total_boxes} defect boxes.")
    print(f"Positive images: {total_positive}; Negative images: {len(samples) - total_positive}")
    for split, split_samples in splits.items():
        positive = sum(1 for sample in split_samples if sample.has_defect)
        boxes = sum(len(sample.labels) for sample in split_samples)
        print(f"{split}: images={len(split_samples)}, positive_images={positive}, boxes={boxes}")
    print(f"Dataset written to: {output_root}")
    print(f"Dataset yaml written to: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
