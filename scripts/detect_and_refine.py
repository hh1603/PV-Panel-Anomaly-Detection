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


def import_sam():
    prepare_ultralytics_env()
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise SystemExit("Ultralytics SAM is unavailable. Check your environment and reinstall requirements.") from exc
    return SAM


def result_to_summary(result, image_path: Path) -> dict:
    detections = []
    boxes = result.boxes
    names = result.names

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls.item())
            label = names[class_id] if isinstance(names, list) else names.get(class_id, str(class_id))
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": label,
                    "confidence": round(float(box.conf.item()), 4),
                    "xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                }
            )

    return {
        "image_path": str(image_path),
        "detections": detections,
    }


def save_result_image(result, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotted = result.plot()
    cv2.imwrite(str(output_path), plotted)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO detection and optional SAM refinement on images.")
    parser.add_argument("--weights", required=True, help="Detector weights path, e.g. runs/train/rgb_yolov8s/weights/best.pt")
    parser.add_argument("--source", required=True, help="Image file or directory to process.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, 0, 0,1.")
    parser.add_argument("--project", default="runs/predict", help="Output project directory, relative to repo root.")
    parser.add_argument("--name", default="predict", help="Run name.")
    parser.add_argument("--sam-model", help="Optional SAM checkpoint, e.g. sam_b.pt or mobile_sam.pt")
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        raise SystemExit(f"Source not found: {source_path}")

    YOLO = import_yolo()
    detector = YOLO(args.weights)
    sam_model = None
    if args.sam_model:
        SAM = import_sam()
        sam_model = SAM(args.sam_model)

    output_root = (PROJECT_ROOT / args.project / args.name).resolve()
    detect_dir = output_root / "detect"
    json_dir = output_root / "json"
    sam_dir = output_root / "sam"
    output_root.mkdir(parents=True, exist_ok=True)

    images = list(iter_image_files(source_path))
    if not images:
        raise SystemExit(f"No images found in: {source_path}")

    for image_path in images:
        result = detector.predict(
            source=str(image_path),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        summary = result_to_summary(result, image_path)
        save_result_image(result, detect_dir / f"{image_path.stem}_det{image_path.suffix}")
        write_json(json_dir / f"{image_path.stem}.json", summary)

        if sam_model and summary["detections"]:
            bboxes = [item["xyxy"] for item in summary["detections"]]
            sam_results = sam_model(str(image_path), bboxes=bboxes, verbose=False)
            for index, sam_result in enumerate(sam_results, start=1):
                save_result_image(sam_result, sam_dir / f"{image_path.stem}_sam_{index}{image_path.suffix}")

    print(f"Inference finished. Outputs saved to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
