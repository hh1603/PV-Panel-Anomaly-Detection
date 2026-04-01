from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, prepare_ultralytics_env


def import_yolo():
    prepare_ultralytics_env()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run `pip install -r requirements.txt` first.") from exc
    return YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a YOLO classification model on folder-based datasets.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset root containing train/val/test class folders.")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Classification checkpoint, e.g. yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu, 0.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--name", help="Run name. Defaults to <dataset name>_<model stem>.")
    parser.add_argument("--project", default="runs/cls", help="Output project directory, relative to repo root.")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset cache.")
    args = parser.parse_args()

    data_dir = args.data.resolve()
    if not data_dir.exists():
        raise SystemExit(f"Dataset root not found: {data_dir}")
    for split in ("train", "val"):
        if not (data_dir / split).exists():
            raise SystemExit(f"Missing split directory: {data_dir / split}")

    YOLO = import_yolo()
    model = YOLO(args.model)
    run_name = args.name or f"{data_dir.name}_{Path(args.model).stem}"
    project_dir = (PROJECT_ROOT / args.project).resolve()

    model.train(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        project=str(project_dir),
        name=run_name,
        cache=args.cache,
    )

    print(f"Classification training finished. Check outputs in: {project_dir / run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
