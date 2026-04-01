from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import PROJECT_ROOT, iter_image_files, prepare_ultralytics_env, write_json


def import_yolo():
    prepare_ultralytics_env()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run `pip install -r requirements.txt` first.") from exc
    return YOLO


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

    top1 = predictions[0] if predictions else None
    return {
        "image_path": str(image_path),
        "top1": top1,
        "predictions": predictions,
    }


def save_result_image(result, output_path: Path, fallback_image: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plotted = result.plot()
        cv2.imwrite(str(output_path), plotted)
    except Exception:
        image = cv2.imread(str(fallback_image))
        if image is not None:
            cv2.imwrite(str(output_path), image)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO classification inference on image files.")
    parser.add_argument("--weights", required=True, help="Classification weights path.")
    parser.add_argument("--source", required=True, help="Image file or directory to process.")
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image size.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, 0.")
    parser.add_argument("--topk", type=int, default=3, help="How many classes to keep in the JSON summary.")
    parser.add_argument("--project", default="runs/cls_predict", help="Output project directory, relative to repo root.")
    parser.add_argument("--name", default="predict", help="Run name.")
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        raise SystemExit(f"Source not found: {source_path}")

    YOLO = import_yolo()
    model = YOLO(args.weights)

    output_root = (PROJECT_ROOT / args.project / args.name).resolve()
    image_dir = output_root / "images"
    json_dir = output_root / "json"
    output_root.mkdir(parents=True, exist_ok=True)

    images = list(iter_image_files(source_path))
    if not images:
        raise SystemExit(f"No images found in: {source_path}")

    for image_path in images:
        result = model.predict(source=str(image_path), imgsz=args.imgsz, device=args.device, verbose=False)[0]
        summary = classify_summary(result, image_path, args.topk)
        save_result_image(result, image_dir / f"{image_path.stem}_cls{image_path.suffix}", image_path)
        write_json(json_dir / f"{image_path.stem}.json", summary)

    print(f"Classification inference finished. Outputs saved to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
