from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


def detection_postprocess_config(pipeline: dict[str, Any], args) -> dict[str, Any]:
    defaults = pipeline.get("detect_postprocess", {}) or {}
    return {
        "confidence": args.post_conf if getattr(args, "post_conf", None) is not None else defaults.get("confidence"),
        "min_area_ratio": args.min_area if getattr(args, "min_area", None) is not None else defaults.get("min_area_ratio"),
        "max_area_ratio": args.max_area if getattr(args, "max_area", None) is not None else defaults.get("max_area_ratio"),
        "max_detections": args.max_detections if getattr(args, "max_detections", None) is not None else defaults.get("max_detections"),
    }


def extract_detections(result, image_path: Path, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config or {}
    boxes = result.boxes
    names = result.names
    detections: list[dict[str, Any]] = []
    if boxes is None:
        return detections

    confidence_threshold = config.get("confidence")
    min_area_ratio = config.get("min_area_ratio")
    max_area_ratio = config.get("max_area_ratio")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image for detection postprocess: {image_path}")
    height, width = image.shape[:2]

    for box in boxes:
        class_id = int(box.cls.item())
        label = names[class_id] if isinstance(names, list) else names.get(class_id, str(class_id))
        conf = float(box.conf.item())
        xyxy_values = [float(value) for value in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = xyxy_values
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(width * height)

        if confidence_threshold is not None and conf < float(confidence_threshold):
            continue
        if min_area_ratio is not None and area_ratio < float(min_area_ratio):
            continue
        if max_area_ratio is not None and area_ratio > float(max_area_ratio):
            continue

        detections.append(
            {
                "class_id": class_id,
                "class_name": label,
                "confidence": round(conf, 4),
                "xyxy": [round(value, 2) for value in xyxy_values],
                "area_ratio": round(area_ratio, 6),
            }
        )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    max_detections = config.get("max_detections")
    if max_detections is not None:
        detections = detections[: int(max_detections)]
    return detections


def detect_summary(image_path: Path, detections: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"image_path": str(image_path), "detections": detections}
    if config:
        payload["postprocess"] = {key: value for key, value in config.items() if value is not None}
    return payload


def render_detection_image(image_path: Path, detections: list[dict[str, Any]]):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image for rendering: {image_path}")

    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["xyxy"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (28, 111, 220), 2)
        caption = f"{detection['class_name']} {detection['confidence']:.3f}"
        (_, text_height), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_top = max(0, y1 - text_height - baseline - 6)
        label_bottom = max(text_height + baseline + 6, y1)
        cv2.rectangle(image, (x1, label_top), (x1 + max(120, len(caption) * 10), label_bottom), (28, 111, 220), -1)
        cv2.putText(
            image,
            caption,
            (x1 + 4, label_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image
