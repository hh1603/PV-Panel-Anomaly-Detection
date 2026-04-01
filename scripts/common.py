from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODALITIES = {"rgb", "ir"}


def prepare_ultralytics_env() -> None:
    ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))


def resolve_modality(value: str) -> str:
    modality = value.strip().lower()
    if modality not in MODALITIES:
        raise ValueError(f"Unsupported modality: {value}. Expected one of: {sorted(MODALITIES)}")
    return modality


def dataset_yaml_for(modality: str) -> Path:
    return CONFIG_DIR / f"{resolve_modality(modality)}_dataset.yaml"


def resolve_dataset_yaml(modality: str | None = None, data: Path | None = None) -> Path:
    if data is not None:
        return data.resolve()
    if modality is not None:
        return dataset_yaml_for(modality).resolve()
    raise ValueError("Either modality or data path must be provided.")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def class_names_from_dataset_yaml(path: Path) -> List[str]:
    data = load_yaml(path)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    if isinstance(names, list):
        return names
    raise ValueError(f"Invalid names section in dataset yaml: {path}")


def iter_image_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            yield source
        return

    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
