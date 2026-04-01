from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image

from common import PROJECT_ROOT, load_yaml, write_json
from run_pipeline import resolve_path


def load_label_file(path: Path) -> list[list[float]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        rows.append([float(value) for value in parts])
    return rows


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * q)))
    return ordered[index]


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def split_dataset_root(data_yaml: Path) -> tuple[Path, dict[str, Any]]:
    payload = load_yaml(data_yaml)
    dataset_root = resolve_path(payload["path"])
    if dataset_root is None or not dataset_root.exists():
        raise SystemExit(f"Dataset root not found from yaml: {data_yaml}")
    return dataset_root, payload


def bucket_name(area_ratio: float) -> str:
    if area_ratio < 0.005:
        return "tiny_lt_0.5pct"
    if area_ratio < 0.02:
        return "small_0.5_to_2pct"
    if area_ratio < 0.05:
        return "medium_2_to_5pct"
    return "large_ge_5pct"


def analyze_split(dataset_root: Path, image_subdir: str, label_subdir: str) -> dict[str, Any]:
    image_dir = dataset_root / image_subdir
    label_dir = dataset_root / label_subdir
    images = sorted([path for path in image_dir.glob("*") if path.is_file()])

    image_count = 0
    positive_images = 0
    negative_images = 0
    total_boxes = 0
    class_counter: Counter[int] = Counter()
    image_sizes: list[tuple[int, int]] = []
    area_ratios: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    x_centers: list[float] = []
    y_centers: list[float] = []
    boxes_per_positive: list[int] = []
    area_buckets: Counter[str] = Counter()

    for image_path in images:
        image_count += 1
        with Image.open(image_path) as image:
            width_px, height_px = image.size
        image_sizes.append((width_px, height_px))

        label_path = label_dir / f"{image_path.stem}.txt"
        labels = load_label_file(label_path)
        if labels:
            positive_images += 1
            boxes_per_positive.append(len(labels))
        else:
            negative_images += 1

        for class_id, x_center, y_center, box_w, box_h in labels:
            total_boxes += 1
            class_counter[int(class_id)] += 1
            area_ratio = box_w * box_h
            area_ratios.append(area_ratio)
            widths.append(box_w)
            heights.append(box_h)
            x_centers.append(x_center)
            y_centers.append(y_center)
            area_buckets[bucket_name(area_ratio)] += 1

    unique_sizes = sorted({f"{w}x{h}" for w, h in image_sizes})
    summary = {
        "images": image_count,
        "positive_images": positive_images,
        "negative_images": negative_images,
        "positive_ratio": ratio(positive_images, image_count),
        "boxes": total_boxes,
        "avg_boxes_per_positive_image": round(mean(boxes_per_positive), 6) if boxes_per_positive else 0.0,
        "class_counts": {str(key): value for key, value in sorted(class_counter.items())},
        "image_sizes": unique_sizes,
        "box_area_ratio": {
            "min": quantile(area_ratios, 0.0),
            "p25": quantile(area_ratios, 0.25),
            "median": quantile(area_ratios, 0.5),
            "p75": quantile(area_ratios, 0.75),
            "max": quantile(area_ratios, 1.0),
            "mean": round(mean(area_ratios), 6) if area_ratios else None,
        },
        "box_width_ratio": {
            "min": quantile(widths, 0.0),
            "median": quantile(widths, 0.5),
            "max": quantile(widths, 1.0),
            "mean": round(mean(widths), 6) if widths else None,
        },
        "box_height_ratio": {
            "min": quantile(heights, 0.0),
            "median": quantile(heights, 0.5),
            "max": quantile(heights, 1.0),
            "mean": round(mean(heights), 6) if heights else None,
        },
        "box_center_ratio": {
            "x_mean": round(mean(x_centers), 6) if x_centers else None,
            "x_median": round(median(x_centers), 6) if x_centers else None,
            "y_mean": round(mean(y_centers), 6) if y_centers else None,
            "y_median": round(median(y_centers), 6) if y_centers else None,
        },
        "area_buckets": dict(area_buckets),
    }
    return summary


def summary_to_markdown(name: str, dataset_root: Path, splits: dict[str, Any]) -> str:
    lines = [f"# {name} Detection Dataset Analysis", ""]
    lines.append(f"- dataset_root: `{dataset_root}`")
    lines.append("")
    for split_name, stats in splits.items():
        lines.append(f"## {split_name}")
        lines.append(f"- images: `{stats['images']}`")
        lines.append(f"- positive_images: `{stats['positive_images']}`")
        lines.append(f"- negative_images: `{stats['negative_images']}`")
        lines.append(f"- positive_ratio: `{stats['positive_ratio']}`")
        lines.append(f"- boxes: `{stats['boxes']}`")
        lines.append(f"- avg_boxes_per_positive_image: `{stats['avg_boxes_per_positive_image']}`")
        lines.append(f"- image_sizes: `{', '.join(stats['image_sizes'])}`")
        lines.append(f"- class_counts: `{stats['class_counts']}`")
        lines.append(f"- area_buckets: `{stats['area_buckets']}`")
        lines.append(f"- box_area_ratio: `{stats['box_area_ratio']}`")
        lines.append(f"- box_width_ratio: `{stats['box_width_ratio']}`")
        lines.append(f"- box_height_ratio: `{stats['box_height_ratio']}`")
        lines.append(f"- box_center_ratio: `{stats['box_center_ratio']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a YOLO detection dataset and output imbalance / box statistics.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "configs" / "dataset_1_defect.yaml", help="Dataset YAML path")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "analysis", help="Output project directory")
    parser.add_argument("--name", default="dataset_1_defect_analysis", help="Output run name")
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        raise SystemExit(f"Dataset yaml not found: {data_yaml}")

    dataset_root, payload = split_dataset_root(data_yaml)
    splits = {}
    for split_name in ("train", "val", "test"):
        image_subdir = payload.get(split_name)
        if not image_subdir:
            continue
        label_subdir = image_subdir.replace("images", "labels", 1)
        splits[split_name] = analyze_split(dataset_root, image_subdir, label_subdir)

    output_dir = args.project.resolve() / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "splits": splits,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(summary_to_markdown(args.name, dataset_root, splits), encoding="utf-8")

    print(f"Analysis finished. Summary saved to: {output_dir / 'summary.json'}")
    for split_name, stats in splits.items():
        print(
            f"{split_name}: images={stats['images']}, positive_images={stats['positive_images']}, "
            f"boxes={stats['boxes']}, positive_ratio={stats['positive_ratio']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
