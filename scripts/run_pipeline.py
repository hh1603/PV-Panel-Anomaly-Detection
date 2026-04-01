from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import PROJECT_ROOT, iter_image_files, load_yaml, prepare_ultralytics_env, write_json
from detect_postprocess import (
    detect_summary,
    detection_postprocess_config,
    extract_detections,
    render_detection_image,
)

REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_registry.yaml"


def import_yolo():
    prepare_ultralytics_env()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run `pip install -r requirements.txt` first.") from exc
    return YOLO


def import_sam():
    prepare_ultralytics_env()
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise SystemExit("Ultralytics SAM is unavailable. Check your environment and reinstall requirements.") from exc
    return SAM


def resolve_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_pipeline_config(name: str) -> dict:
    registry = load_yaml(REGISTRY_PATH)
    pipelines = registry.get("pipelines", {})
    if name not in pipelines:
        raise SystemExit(f"Unknown pipeline: {name}. Available: {sorted(pipelines)}")
    return pipelines[name]


def classify_summary(result, image_path: Path, topk: int) -> dict:
    probs = result.probs
    if probs is None:
        return {"image_path": str(image_path), "predictions": [], "top1": None}
    scores = probs.data.cpu().tolist()
    names = result.names
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:topk]
    predictions = []
    for class_id, score in ranked:
        label = names[class_id] if isinstance(names, list) else names.get(class_id, str(class_id))
        predictions.append(
            {
                "class_id": class_id,
                "class_name": label,
                "confidence": round(float(score), 4),
            }
        )
    return {"image_path": str(image_path), "top1": predictions[0] if predictions else None, "predictions": predictions}


def save_result_image(result, output_path: Path, fallback_image: Path | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plotted = result.plot()
        cv2.imwrite(str(output_path), plotted)
    except Exception:
        if fallback_image is not None:
            image = cv2.imread(str(fallback_image))
            if image is not None:
                cv2.imwrite(str(output_path), image)


def train_pipeline(args) -> int:
    pipeline = load_pipeline_config(args.pipeline)
    defaults = pipeline.get("default_train", {})
    data_path = resolve_path(pipeline["data"])
    YOLO = import_yolo()

    model_path = args.model or defaults.get("model")
    if not model_path:
        raise SystemExit("No model checkpoint configured or provided.")

    model = YOLO(model_path)
    project = resolve_path(args.project or defaults.get("project") or "runs")
    name = args.name or defaults.get("name") or f"{args.pipeline}_{Path(model_path).stem}"
    epochs = args.epochs if args.epochs is not None else defaults.get("epochs", 50)
    imgsz = args.imgsz if args.imgsz is not None else defaults.get("imgsz", 640)
    batch = args.batch if args.batch is not None else defaults.get("batch", 16)
    workers = args.workers if args.workers is not None else defaults.get("workers", 4)
    patience = args.patience if args.patience is not None else defaults.get("patience", 20)

    train_kwargs = dict(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=args.device,
        patience=patience,
        project=str(project),
        name=name,
        cache=args.cache,
    )

    model.train(**train_kwargs)
    print(f"Pipeline training finished. Outputs saved to: {project / name}")
    return 0


def predict_pipeline(args) -> int:
    pipeline = load_pipeline_config(args.pipeline)
    task = pipeline["task"]
    defaults = pipeline.get("default_predict", {})
    weights = resolve_path(args.weights or defaults.get("weights"))
    if weights is None or not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    source = Path(args.source).resolve()
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    YOLO = import_yolo()
    model = YOLO(str(weights))
    project = resolve_path(args.project or defaults.get("project") or "runs/predict")
    name = args.name or defaults.get("name") or f"{args.pipeline}_predict"
    output_root = (project / name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    images = list(iter_image_files(source))
    if not images:
        raise SystemExit(f"No images found in: {source}")

    if task == "detect":
        conf = args.conf if args.conf is not None else defaults.get("conf", 0.25)
        imgsz = args.imgsz if args.imgsz is not None else defaults.get("imgsz", 640)
        postprocess = detection_postprocess_config(pipeline, args)
        sam_model = None
        if args.sam_model:
            SAM = import_sam()
            sam_model = SAM(args.sam_model)
        detect_dir = output_root / "detect"
        json_dir = output_root / "json"
        sam_dir = output_root / "sam"
        for image_path in images:
            result = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, device=args.device, verbose=False)[0]
            detections = extract_detections(result, image_path, postprocess)
            summary = detect_summary(image_path, detections, postprocess)
            rendered = render_detection_image(image_path, detections)
            detect_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(detect_dir / f"{image_path.stem}_det{image_path.suffix}"), rendered)
            write_json(json_dir / f"{image_path.stem}.json", summary)
            if sam_model and detections:
                bboxes = [item["xyxy"] for item in detections]
                sam_results = sam_model(str(image_path), bboxes=bboxes, verbose=False)
                for index, sam_result in enumerate(sam_results, start=1):
                    save_result_image(sam_result, sam_dir / f"{image_path.stem}_sam_{index}{image_path.suffix}", image_path)
    elif task == "classify":
        imgsz = args.imgsz if args.imgsz is not None else defaults.get("imgsz", 224)
        topk = args.topk if args.topk is not None else defaults.get("topk", 3)
        image_dir = output_root / "images"
        json_dir = output_root / "json"
        for image_path in images:
            result = model.predict(source=str(image_path), imgsz=imgsz, device=args.device, verbose=False)[0]
            summary = classify_summary(result, image_path, topk)
            save_result_image(result, image_dir / f"{image_path.stem}_cls{image_path.suffix}", image_path)
            write_json(json_dir / f"{image_path.stem}.json", summary)
    else:
        raise SystemExit(f"Unsupported pipeline task: {task}")

    print(f"Pipeline inference finished. Outputs saved to: {output_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run named visible/infrared pipelines using the project registry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a named pipeline.")
    train_parser.add_argument("--pipeline", required=True, help="Pipeline name from configs/model_registry.yaml")
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

    predict_parser = subparsers.add_parser("predict", help="Run inference for a named pipeline.")
    predict_parser.add_argument("--pipeline", required=True, help="Pipeline name from configs/model_registry.yaml")
    predict_parser.add_argument("--weights", help="Override weights path.")
    predict_parser.add_argument("--source", required=True, help="Image file or directory.")
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
        return train_pipeline(args)
    if args.command == "predict":
        return predict_pipeline(args)
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
