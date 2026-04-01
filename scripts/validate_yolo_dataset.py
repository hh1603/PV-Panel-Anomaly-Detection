from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import PROJECT_ROOT, class_names_from_dataset_yaml, load_yaml, resolve_dataset_yaml

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def validate_label_file(label_path: Path, class_count: int) -> tuple[list[str], Counter]:
    errors: list[str] = []
    counts: Counter = Counter()

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}: line {line_number} should contain 5 values, got {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}: line {line_number} contains non-numeric values")
            continue

        if class_id < 0 or class_id >= class_count:
            errors.append(f"{label_path}: line {line_number} has class id {class_id} outside [0, {class_count - 1}]")

        if any(value < 0 or value > 1 for value in coords):
            errors.append(f"{label_path}: line {line_number} has coordinates outside [0, 1]")

        width, height = coords[2], coords[3]
        if width <= 0 or height <= 0:
            errors.append(f"{label_path}: line {line_number} has non-positive width or height")

        counts[class_id] += 1

    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset layout and label format.")
    parser.add_argument("--modality", choices=["rgb", "ir"], help="Dataset modality to validate using the default config.")
    parser.add_argument("--data", type=Path, help="Explicit dataset yaml path.")
    args = parser.parse_args()

    try:
        dataset_yaml = resolve_dataset_yaml(args.modality, args.data)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not dataset_yaml.exists():
        raise SystemExit(f"Dataset yaml not found: {dataset_yaml}")

    dataset = load_yaml(dataset_yaml)
    class_names = class_names_from_dataset_yaml(dataset_yaml)
    dataset_root = (PROJECT_ROOT / dataset["path"]).resolve()

    all_errors: list[str] = []
    class_counter: Counter = Counter()
    image_count = 0
    missing_label_files: list[Path] = []
    orphan_label_files: list[Path] = []

    for split in SPLITS:
        image_dir = dataset_root / dataset[split]
        label_dir = dataset_root / "labels" / split

        if not image_dir.exists():
            all_errors.append(f"Missing image directory: {image_dir}")
            continue
        if not label_dir.exists():
            all_errors.append(f"Missing label directory: {label_dir}")
            continue

        images = [path for path in sorted(image_dir.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        labels = {path.stem: path for path in label_dir.rglob("*.txt") if path.is_file()}

        image_count += len(images)

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_label_files.append(label_path)
                continue

            errors, counts = validate_label_file(label_path, len(class_names))
            all_errors.extend(errors)
            class_counter.update(counts)

        image_stems = {image_path.stem for image_path in images}
        orphan_label_files.extend(path for stem, path in labels.items() if stem not in image_stems)

    print(f"Dataset yaml: {dataset_yaml}")
    print(f"Dataset root: {dataset_root}")
    print(f"Total images scanned: {image_count}")
    print("Class distribution:")
    for index, name in enumerate(class_names):
        print(f"  {index}: {name} -> {class_counter.get(index, 0)} boxes")

    if missing_label_files:
        print(f"Warning: {len(missing_label_files)} images do not have matching label files.")
        for path in missing_label_files[:20]:
            print(f"  missing: {path}")
        if len(missing_label_files) > 20:
            print("  ...")

    if orphan_label_files:
        print(f"Warning: {len(orphan_label_files)} label files do not have matching images.")
        for path in orphan_label_files[:20]:
            print(f"  orphan: {path}")
        if len(orphan_label_files) > 20:
            print("  ...")

    if all_errors:
        print(f"Validation failed with {len(all_errors)} errors:")
        for error in all_errors[:50]:
            print(f"  {error}")
        if len(all_errors) > 50:
            print("  ...")
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
