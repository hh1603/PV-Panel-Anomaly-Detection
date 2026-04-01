from __future__ import annotations

import argparse

from common import load_yaml
from run_pipeline import REGISTRY_PATH, predict_pipeline, train_pipeline

MODALITY_ALIASES = {
    "visible": "visible",
    "rgb": "visible",
    "infrared": "infrared",
    "ir": "infrared",
}


def resolve_modality(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in MODALITY_ALIASES:
        raise SystemExit(
            f"Unsupported modality: {value}. Expected one of: {sorted(MODALITY_ALIASES)}"
        )
    return MODALITY_ALIASES[normalized]


def resolve_pipeline_name(modality: str, command: str) -> str:
    registry = load_yaml(REGISTRY_PATH)
    routes = registry.get("modality_routes", {})
    normalized = resolve_modality(modality)
    route = routes.get(normalized)
    if not route:
        raise SystemExit(f"No modality route configured for: {normalized}")
    pipeline_name = route.get(command)
    if not pipeline_name:
        raise SystemExit(f"No pipeline route configured for modality={normalized}, command={command}")
    return pipeline_name


def run_train(args) -> int:
    pipeline_name = resolve_pipeline_name(args.modality, "train")
    forwarded = argparse.Namespace(
        pipeline=pipeline_name,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        name=args.name,
        project=args.project,
        cache=args.cache,
    )
    print(f"Resolved modality '{resolve_modality(args.modality)}' to training pipeline '{pipeline_name}'.")
    return train_pipeline(forwarded)


def run_predict(args) -> int:
    pipeline_name = resolve_pipeline_name(args.modality, "predict")
    forwarded = argparse.Namespace(
        pipeline=pipeline_name,
        weights=args.weights,
        source=args.source,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        post_conf=args.post_conf,
        min_area=args.min_area,
        max_area=args.max_area,
        max_detections=args.max_detections,
        topk=args.topk,
        name=args.name,
        project=args.project,
        sam_model=args.sam_model,
    )
    print(f"Resolved modality '{resolve_modality(args.modality)}' to inference pipeline '{pipeline_name}'.")
    return predict_pipeline(forwarded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run project training or inference by modality without remembering pipeline names."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the default pipeline for a modality.")
    train_parser.add_argument("--modality", required=True, help="visible/rgb or infrared/ir")
    train_parser.add_argument("--model", help="Override model checkpoint.")
    train_parser.add_argument("--epochs", type=int, help="Override epochs.")
    train_parser.add_argument("--imgsz", type=int, help="Override image size.")
    train_parser.add_argument("--batch", type=int, help="Override batch size.")
    train_parser.add_argument("--workers", type=int, help="Override workers.")
    train_parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu, 0.")
    train_parser.add_argument("--patience", type=int, help="Override early stopping patience.")
    train_parser.add_argument("--name", help="Run name override.")
    train_parser.add_argument("--project", help="Output project override.")
    train_parser.add_argument("--cache", action="store_true", help="Enable Ultralytics cache.")

    predict_parser = subparsers.add_parser("predict", help="Run inference for the default pipeline of a modality.")
    predict_parser.add_argument("--modality", required=True, help="visible/rgb or infrared/ir")
    predict_parser.add_argument("--source", required=True, help="Image file or directory.")
    predict_parser.add_argument("--weights", help="Override weights path.")
    predict_parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, 0.")
    predict_parser.add_argument("--imgsz", type=int, help="Override image size.")
    predict_parser.add_argument("--conf", type=float, help="Base detection confidence threshold passed to YOLO.")
    predict_parser.add_argument("--post-conf", type=float, help="Detection confidence threshold applied after prediction.")
    predict_parser.add_argument("--min-area", type=float, help="Minimum normalized box area kept after prediction.")
    predict_parser.add_argument("--max-area", type=float, help="Maximum normalized box area kept after prediction.")
    predict_parser.add_argument("--max-detections", type=int, help="Maximum number of detections to keep after filtering.")
    predict_parser.add_argument("--topk", type=int, help="Classification top-k to keep.")
    predict_parser.add_argument("--name", help="Run name override.")
    predict_parser.add_argument("--project", help="Output project override.")
    predict_parser.add_argument("--sam-model", help="Optional SAM checkpoint for detection pipelines.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        return run_train(args)
    if args.command == "predict":
        return run_predict(args)
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
