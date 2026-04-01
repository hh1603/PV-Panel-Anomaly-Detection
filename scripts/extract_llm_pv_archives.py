from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile

from common import PROJECT_ROOT

ARCHIVES = {
    "visible_binary": PROJECT_ROOT / "LLM-PV-image-database" / "PV-visible-binary.zip",
    "visible_multiclass": PROJECT_ROOT / "LLM-PV-image-database" / "PV-visible-multiclass.zip",
    "ir_binary": PROJECT_ROOT / "LLM-PV-image-database" / "PV-IR-binary.zip",
    "ir_multiclass": PROJECT_ROOT / "LLM-PV-image-database" / "PV-IR-multiclass.zip",
    "el_binary": PROJECT_ROOT / "LLM-PV-image-database" / "PV-EL-binary.zip",
    "el_multiclass": PROJECT_ROOT / "LLM-PV-image-database" / "PV-EL-multiclass.zip",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def remove_hidden_files(directory: Path) -> None:
    for path in directory.rglob('*'):
        if path.is_file() and path.name.startswith('.'):
            path.unlink()


def clean_macosx(directory: Path) -> None:
    macosx_dir = directory / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)


def flatten_single_root(directory: Path) -> None:
    entries = [path for path in directory.iterdir() if path.name != "__MACOSX"]
    if len(entries) != 1 or not entries[0].is_dir():
        return

    root_dir = entries[0]
    for item in list(root_dir.iterdir()):
        shutil.move(str(item), directory / item.name)
    root_dir.rmdir()


def extract_archive(archive_path: Path, target_dir: Path, overwrite: bool) -> tuple[int, int]:
    if target_dir.exists() and overwrite:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive_path) as archive:
        archive.extractall(target_dir)

    clean_macosx(target_dir)
    flatten_single_root(target_dir)
    remove_hidden_files(target_dir)

    files = [path for path in target_dir.rglob("*") if path.is_file()]
    image_files = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    return len(files), len(image_files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract LLM PV archives into clear visible/ir/el dataset folders.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "classification_sources", help="Output root directory.")
    parser.add_argument("--name", choices=sorted(ARCHIVES), nargs="*", help="Specific archive names to extract. Defaults to all.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing extracted folders.")
    args = parser.parse_args()

    selected_names = args.name or list(ARCHIVES.keys())
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for name in selected_names:
        archive_path = ARCHIVES[name]
        if not archive_path.exists():
            print(f"Skip {name}: archive not found -> {archive_path}")
            continue
        target_dir = output_root / name
        file_count, image_count = extract_archive(archive_path, target_dir, overwrite=args.overwrite)
        print(f"{name}: files={file_count}, images={image_count}, output={target_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
