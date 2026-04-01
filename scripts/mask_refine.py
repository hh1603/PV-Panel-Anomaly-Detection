from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import PROJECT_ROOT, prepare_ultralytics_env

KNOWN_SAM_CHECKPOINTS = (
    "mobile_sam.pt",
    "sam_b.pt",
    "sam_l.pt",
    "sam_h.pt",
    "sam2_t.pt",
    "sam2_s.pt",
    "sam2_b.pt",
)
SEARCH_DIRS = ("weights", "models", "checkpoints", "assets")
SAM_CACHE: dict[tuple[str, str], Any] = {}
MASK_COLORS = [
    (28, 111, 220),
    (74, 176, 106),
    (232, 146, 34),
    (176, 86, 196),
    (60, 189, 198),
]


def import_sam():
    prepare_ultralytics_env()
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise RuntimeError("Ultralytics SAM is unavailable in this environment.") from exc
    return SAM


def discover_sam_checkpoint() -> Path | None:
    env_path = os.environ.get("SOLAR_SAM_CHECKPOINT")
    if env_path:
        candidate = Path(env_path).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if candidate.exists():
            return candidate

    for dirname in SEARCH_DIRS:
        directory = PROJECT_ROOT / dirname
        if not directory.exists():
            continue
        for filename in KNOWN_SAM_CHECKPOINTS:
            candidate = directory / filename
            if candidate.exists():
                return candidate.resolve()
        for candidate in directory.rglob("*.pt"):
            lower = candidate.name.lower()
            if lower.startswith(("sam", "mobile_sam", "fastsam")):
                return candidate.resolve()

    for filename in KNOWN_SAM_CHECKPOINTS:
        candidate = PROJECT_ROOT / filename
        if candidate.exists():
            return candidate.resolve()
    return None


def get_sam_model(checkpoint: Path, device: str) -> Any:
    cache_key = (str(checkpoint), device)
    model = SAM_CACHE.get(cache_key)
    if model is None:
        SAM = import_sam()
        model = SAM(str(checkpoint))
        SAM_CACHE[cache_key] = model
    return model


def grid_position(cx: float, cy: float, width: int, height: int) -> str:
    x_ratio = cx / max(width, 1)
    y_ratio = cy / max(height, 1)

    if x_ratio < 1 / 3:
        x_label = "left"
    elif x_ratio < 2 / 3:
        x_label = "center"
    else:
        x_label = "right"

    if y_ratio < 1 / 3:
        y_label = "top"
    elif y_ratio < 2 / 3:
        y_label = "middle"
    else:
        y_label = "bottom"

    return f"{y_label}-{x_label}"


def size_bucket(area_ratio: float) -> str:
    if area_ratio < 0.005:
        return "tiny"
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def simplify_polygon(points: np.ndarray) -> list[list[float]]:
    if points.size == 0:
        return []
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    epsilon = max(2.0, 0.01 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in approx.tolist()]


def derive_shape(mask_pixels: int, bbox: list[float]) -> str:
    x1, y1, x2, y2 = bbox
    box_w = max(float(x2 - x1), 1.0)
    box_h = max(float(y2 - y1), 1.0)
    fill_ratio = mask_pixels / max(box_w * box_h, 1.0)
    aspect_ratio = max(box_w / box_h, box_h / box_w)

    if aspect_ratio >= 3.0:
        return "elongated"
    if fill_ratio >= 0.7:
        return "blocky"
    if fill_ratio <= 0.35:
        return "sparse"
    return "irregular"


def region_from_mask(binary_mask: np.ndarray, detection: dict[str, Any], image_shape: tuple[int, int], polygon: np.ndarray | None = None) -> tuple[dict[str, Any], np.ndarray] | None:
    mask = (binary_mask > 0).astype(np.uint8)
    pixels = int(mask.sum())
    if pixels == 0:
        return None

    height, width = image_shape
    mask_area_ratio = pixels / float(max(height * width, 1))

    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"]:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        ys, xs = np.where(mask > 0)
        cx = float(xs.mean())
        cy = float(ys.mean())

    if polygon is None or len(polygon) == 0:
        polygon_points = contour.reshape(-1, 2)
    else:
        polygon_points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)

    x, y, w_box, h_box = cv2.boundingRect(contour)
    bbox = [round(float(x), 2), round(float(y), 2), round(float(x + w_box), 2), round(float(y + h_box), 2)]

    summary = {
        "class_id": int(detection.get("class_id", 0)),
        "class_name": str(detection.get("class_name", "defect")),
        "confidence": round(float(detection.get("confidence", 0.0)), 4),
        "bbox": [round(float(value), 2) for value in detection.get("xyxy", bbox)],
        "mask_bbox": bbox,
        "bbox_area_ratio": round(float(detection.get("area_ratio", 0.0)), 6),
        "mask_pixels": pixels,
        "mask_area_ratio": round(mask_area_ratio, 6),
        "center": [round(cx, 2), round(cy, 2)],
        "position": grid_position(cx, cy, width, height),
        "size": size_bucket(mask_area_ratio),
        "shape": derive_shape(pixels, bbox),
        "polygon": simplify_polygon(polygon_points),
        "contour_points": int(len(polygon_points)),
    }
    return summary, mask


def grabcut_mask(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        if int(binary.sum()) == 0:
            raise RuntimeError("empty grabcut mask")
        return binary
    except Exception:
        fallback = np.zeros(image.shape[:2], np.uint8)
        fallback[y1:y2, x1:x2] = 1
        return fallback


def collect_regions_from_sam(image: np.ndarray, detections: list[dict[str, Any]], checkpoint: Path, device: str) -> list[tuple[dict[str, Any], np.ndarray]]:
    if not detections:
        return []
    model = get_sam_model(checkpoint, device)
    results = model(str(PROJECT_ROOT / "__unused__"), verbose=False)
    raise RuntimeError("SAM placeholder should have been replaced")


def run_sam_refinement(image_path: Path, image: np.ndarray, detections: list[dict[str, Any]], checkpoint: Path, device: str) -> list[tuple[dict[str, Any], np.ndarray]]:
    model = get_sam_model(checkpoint, device)
    sam_results = model(str(image_path), bboxes=[item["xyxy"] for item in detections], verbose=False)
    regions: list[tuple[dict[str, Any], np.ndarray]] = []
    detection_index = 0

    for sam_result in sam_results:
        masks = getattr(sam_result, "masks", None)
        if masks is None:
            continue
        mask_data = masks.data.cpu().numpy()
        polygons = masks.xy
        for idx in range(mask_data.shape[0]):
            detection = detections[min(detection_index, len(detections) - 1)]
            polygon = polygons[idx] if idx < len(polygons) else None
            region = region_from_mask(mask_data[idx] > 0.5, detection, image.shape[:2], polygon)
            detection_index += 1
            if region is not None:
                regions.append(region)
    return regions


def run_grabcut_refinement(image: np.ndarray, detections: list[dict[str, Any]]) -> list[tuple[dict[str, Any], np.ndarray]]:
    regions: list[tuple[dict[str, Any], np.ndarray]] = []
    for detection in detections:
        binary = grabcut_mask(image, detection["xyxy"])
        region = region_from_mask(binary, detection, image.shape[:2])
        if region is not None:
            regions.append(region)
    return regions


def render_mask_outputs(image: np.ndarray, regions: list[tuple[dict[str, Any], np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    overlay = image.copy()
    mask_only = np.zeros_like(image)

    for index, (region, binary_mask) in enumerate(regions):
        color = MASK_COLORS[index % len(MASK_COLORS)]
        mask = binary_mask.astype(bool)
        color_layer = np.zeros_like(image)
        color_layer[mask] = color
        overlay[mask] = cv2.addWeighted(overlay, 0.45, color_layer, 0.55, 0)[mask]
        mask_only[mask] = color

        polygon = np.asarray(region.get("polygon", []), dtype=np.float32)
        if polygon.size:
            contour = polygon.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(overlay, [contour], True, color, 2)
            cv2.polylines(mask_only, [contour], True, (255, 255, 255), 1)

        x1, y1, _, _ = [int(round(v)) for v in region["mask_bbox"]]
        label = f"{region['class_name']} {region['mask_area_ratio'] * 100:.2f}%"
        (_, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_top = max(0, y1 - text_height - baseline - 6)
        label_bottom = max(text_height + baseline + 6, y1)
        cv2.rectangle(overlay, (x1, label_top), (x1 + max(140, len(label) * 10), label_bottom), color, -1)
        cv2.putText(
            overlay,
            label,
            (x1 + 4, label_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay, mask_only


def refine_detection_masks(
    image_path: Path,
    detections: list[dict[str, Any]],
    device: str = "cpu",
    sam_checkpoint: Path | None = None,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image for mask refinement: {image_path}")

    if not detections:
        return {
            "summary": {
                "enabled": False,
                "strategy": "none",
                "checkpoint": None,
                "message": "No detections were kept, so no mask refinement was run.",
                "region_count": 0,
                "regions": [],
            },
            "overlay_image": None,
            "mask_only_image": None,
        }

    checkpoint = sam_checkpoint or discover_sam_checkpoint()
    regions: list[tuple[dict[str, Any], np.ndarray]] = []
    strategy = "grabcut"
    message = "No SAM checkpoint found. Used GrabCut fallback refinement."

    if checkpoint is not None:
        try:
            regions = run_sam_refinement(image_path, image, detections, checkpoint, device)
            strategy = "sam"
            message = f"Used SAM refinement with checkpoint: {checkpoint.name}"
        except Exception as exc:
            regions = []
            strategy = "grabcut"
            message = f"SAM refinement failed ({exc}). Used GrabCut fallback refinement."

    if not regions:
        regions = run_grabcut_refinement(image, detections)
        if checkpoint is not None and "SAM refinement failed" not in message:
            message = f"SAM checkpoint was found ({checkpoint.name}), but no mask was returned. Used GrabCut fallback refinement."

    overlay_image, mask_only_image = render_mask_outputs(image, regions)
    region_summaries = [item[0] for item in regions]

    return {
        "summary": {
            "enabled": True,
            "strategy": strategy,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "message": message,
            "region_count": len(region_summaries),
            "regions": region_summaries,
        },
        "overlay_image": overlay_image,
        "mask_only_image": mask_only_image,
    }
