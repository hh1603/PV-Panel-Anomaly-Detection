from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from common import PROJECT_ROOT

DATASET_CONFIG = {
    "visible_binary": {
        "source": PROJECT_ROOT / "datasets" / "classification_sources" / "visible_binary",
        "output": PROJECT_ROOT / "datasets" / "classification_ready" / "visible_binary",
        "mode": "binary",
    },
    "visible_multiclass": {
        "source": PROJECT_ROOT / "datasets" / "classification_sources" / "visible_multiclass",
        "output": PROJECT_ROOT / "datasets" / "classification_ready" / "visible_multiclass",
        "mode": "multiclass",
    },
    "ir_binary": {
        "source": PROJECT_ROOT / "datasets" / "classification_sources" / "ir_binary",
        "output": PROJECT_ROOT / "datasets" / "classification_ready" / "ir_binary",
        "mode": "binary",
    },
    "ir_multiclass": {
        "source": PROJECT_ROOT / "datasets" / "classification_sources" / "ir_multiclass",
        "output": PROJECT_ROOT / "datasets" / "classification_ready" / "ir_multiclass",
        "mode": "multiclass",
    },
    "el_binary": {
        "source": PROJECT_ROOT / "datasets" / "classification_sources" / "el_binary",
        "output": PROJECT_ROOT / "datasets" / "classification_ready" / "el_binary",
        "mode": "binary",
    },
    "el_multiclass": {
        "source": PROJECT_ROOT / "datasets" / "classification_sources" / "el_multiclass",
        "output": PROJECT_ROOT / "datasets" / "classification_ready" / "el_multiclass",
        "mode": "multiclass",
    },
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sanitize_label(label: str) -> str:
    return label.strip().lower().replace(' ', '_').replace('-', '_')


def parse_raw_label(path: Path) -> str:
    stem = path.stem
    parts = stem.split('_', 1)
    if len(parts) == 1:
        raise ValueError(f"Cannot parse class label from filename: {path.name}")
    return sanitize_label(parts[1])


def final_label(raw_label: str, mode: str) -> str:
    if mode == 'binary':
        return 'healthy' if raw_label == 'healthy' else 'defect'
    return raw_label


def split_items(items: list[Path], train_ratio: float, val_ratio: float) -> tuple[list[Path], list[Path], list[Path]]:
    train_count = int(len(items) * train_ratio)
    val_count = int(len(items) * val_ratio)
    train_items = items[:train_count]
    val_items = items[train_count:train_count + val_count]
    test_items = items[train_count + val_count:]
    return train_items, val_items, test_items


def copy_split(items: list[Path], split: str, label: str, output_root: Path) -> None:
    target_dir = output_root / split / label
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        shutil.copy2(item, target_dir / item.name)


def prepare_dataset(name: str, train_ratio: float, val_ratio: float, seed: int, overwrite: bool) -> None:
    config = DATASET_CONFIG[name]
    source_root = config['source']
    output_root = config['output']
    mode = config['mode']

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(source_root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        raw_label = parse_raw_label(path)
        grouped[final_label(raw_label, mode)].append(path)

    rng = random.Random(seed)
    summary: dict[str, dict[str, int]] = defaultdict(dict)
    for label, items in grouped.items():
        rng.shuffle(items)
        train_items, val_items, test_items = split_items(items, train_ratio, val_ratio)
        copy_split(train_items, 'train', label, output_root)
        copy_split(val_items, 'val', label, output_root)
        copy_split(test_items, 'test', label, output_root)
        summary[label] = {
            'train': len(train_items),
            'val': len(val_items),
            'test': len(test_items),
        }

    print(f"[{name}] -> {output_root}")
    for label in sorted(summary):
        stats = summary[label]
        print(f"  {label}: train={stats['train']}, val={stats['val']}, test={stats['test']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare visible / IR / EL classification datasets into train/val/test folder layout.")
    parser.add_argument("--name", choices=sorted(DATASET_CONFIG), nargs='*', help="Specific dataset names to prepare. Defaults to all.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--overwrite", action='store_true', help="Replace existing prepared datasets.")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("train_ratio and val_ratio must satisfy: train_ratio > 0, val_ratio >= 0, train_ratio + val_ratio < 1")

    selected_names = args.name or list(DATASET_CONFIG.keys())
    for name in selected_names:
        prepare_dataset(name, args.train_ratio, args.val_ratio, args.seed, args.overwrite)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
