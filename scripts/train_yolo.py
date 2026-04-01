from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, prepare_ultralytics_env, resolve_dataset_yaml


def import_yolo():
    prepare_ultralytics_env()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run `pip install -r requirements.txt` first.") from exc
    return YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for solar panel fault images.")
    parser.add_argument("--modality", choices=["rgb", "ir"], help="Which default detector config to train.")
    parser.add_argument("--data", type=Path, help="Explicit dataset yaml path.")
    parser.add_argument("--model", default="yolov8s.pt", help="Pretrained model checkpoint to start from.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu, 0, 0,1.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--name", help="Run name. Defaults to <dataset stem>_<model stem>.")
    parser.add_argument("--project", default="runs/train", help="Output project directory, relative to repo root.")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset cache.")
    args = parser.parse_args()

    try:
        dataset_yaml = resolve_dataset_yaml(args.modality, args.data)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not dataset_yaml.exists():
        raise SystemExit(f"Dataset yaml not found: {dataset_yaml}")

    YOLO = import_yolo()
    model = YOLO(args.model)
    run_name = args.name or f"{dataset_yaml.stem}_{Path(args.model).stem}"
    project_dir = (PROJECT_ROOT / args.project).resolve()

    model.train(
        data=str(dataset_yaml),
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

    print(f"Training finished. Check outputs in: {project_dir / run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
