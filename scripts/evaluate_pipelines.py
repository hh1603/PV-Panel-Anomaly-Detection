from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import PROJECT_ROOT, load_yaml, prepare_ultralytics_env, write_json
from run_pipeline import REGISTRY_PATH, resolve_path


def import_yolo():
    prepare_ultralytics_env()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run `pip install -r requirements.txt` first.") from exc
    return YOLO


def load_pipeline_config(name: str) -> dict[str, Any]:
    registry = load_yaml(REGISTRY_PATH)
    pipelines = registry.get("pipelines", {})
    if name not in pipelines:
        raise SystemExit(f"Unknown pipeline: {name}. Available: {sorted(pipelines)}")
    return pipelines[name]


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except Exception:
        return None


def metric_value(container: dict[str, Any], key: str) -> float | None:
    return as_float(container.get(key))


def summarize_detect(metrics, pipeline_name: str, weights: Path, data: Path, run_dir: Path) -> dict[str, Any]:
    results = getattr(metrics, "results_dict", {}) or {}
    box = getattr(metrics, "box", None)
    summary = {
        "pipeline": pipeline_name,
        "task": "detect",
        "weights": str(weights),
        "data": str(data),
        "run_dir": str(run_dir),
        "metrics": {
            "precision": metric_value(results, "metrics/precision(B)"),
            "recall": metric_value(results, "metrics/recall(B)"),
            "map50": metric_value(results, "metrics/mAP50(B)"),
            "map50_95": metric_value(results, "metrics/mAP50-95(B)"),
            "fitness": metric_value(results, "fitness"),
        },
        "speed_ms": {k: as_float(v) for k, v in getattr(metrics, "speed", {}).items()},
        "per_class_map50": {},
    }
    if box is not None and hasattr(box, "ap50"):
        names = getattr(metrics, "names", {}) or {}
        per_class = {}
        for index, value in enumerate(list(box.ap50)):
            label = names[index] if isinstance(names, list) else names.get(index, str(index))
            per_class[label] = as_float(value)
        summary["per_class_map50"] = per_class
    return summary


def summarize_classify(metrics, pipeline_name: str, weights: Path, data: Path, run_dir: Path) -> dict[str, Any]:
    results = getattr(metrics, "results_dict", {}) or {}
    top1 = metric_value(results, "metrics/accuracy_top1")
    top5 = metric_value(results, "metrics/accuracy_top5")
    if top1 is None:
        top1 = metric_value(results, "accuracy_top1")
    if top5 is None:
        top5 = metric_value(results, "accuracy_top5")
    summary = {
        "pipeline": pipeline_name,
        "task": "classify",
        "weights": str(weights),
        "data": str(data),
        "run_dir": str(run_dir),
        "metrics": {
            "top1": top1,
            "top5": top5,
            "fitness": metric_value(results, "fitness"),
        },
        "speed_ms": {k: as_float(v) for k, v in getattr(metrics, "speed", {}).items()},
    }
    return summary


def summary_to_markdown(summary: dict[str, Any]) -> str:
    lines = [f"# {summary['pipeline']} Evaluation", ""]
    lines.append(f"- task: `{summary['task']}`")
    lines.append(f"- weights: `{summary['weights']}`")
    lines.append(f"- data: `{summary['data']}`")
    lines.append(f"- run_dir: `{summary['run_dir']}`")
    lines.append("")
    lines.append("## Metrics")
    for key, value in summary.get("metrics", {}).items():
        lines.append(f"- {key}: `{value}`")
    speed = summary.get("speed_ms", {})
    if speed:
        lines.append("")
        lines.append("## Speed (ms)")
        for key, value in speed.items():
            lines.append(f"- {key}: `{value}`")
    per_class = summary.get("per_class_map50", {})
    if per_class:
        lines.append("")
        lines.append("## Per-class mAP50")
        for key, value in per_class.items():
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def evaluate_pipeline(args) -> int:
    pipeline = load_pipeline_config(args.pipeline)
    defaults = pipeline.get("default_predict", {})
    weights = resolve_path(args.weights or defaults.get("weights"))
    if weights is None or not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    data = resolve_path(args.data or pipeline.get("data"))
    if data is None or not data.exists():
        raise SystemExit(f"Data not found: {data}")

    YOLO = import_yolo()
    model = YOLO(str(weights))

    project = resolve_path(args.project or "runs/eval")
    name = args.name or f"{args.pipeline}_eval"
    imgsz = args.imgsz if args.imgsz is not None else defaults.get("imgsz", 640)
    batch = args.batch if args.batch is not None else 1
    workers = args.workers if args.workers is not None else 0
    split = args.split

    val_kwargs = {
        "data": str(data),
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": args.device,
        "project": str(project),
        "name": name,
        "split": split,
        "verbose": False,
    }
    if pipeline["task"] == "detect":
        val_kwargs["conf"] = args.conf if args.conf is not None else defaults.get("conf", 0.25)

    metrics = model.val(**val_kwargs)
    run_dir = (project / name).resolve()

    if pipeline["task"] == "detect":
        summary = summarize_detect(metrics, args.pipeline, weights, data, run_dir)
    elif pipeline["task"] == "classify":
        summary = summarize_classify(metrics, args.pipeline, weights, data, run_dir)
    else:
        raise SystemExit(f"Unsupported task: {pipeline['task']}")

    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(summary_to_markdown(summary), encoding="utf-8")

    print(f"Evaluation finished. Summary saved to: {run_dir / 'summary.json'}")
    for key, value in summary["metrics"].items():
        print(f"{key}={value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a named pipeline using the current registry and weights.")
    parser.add_argument("--pipeline", required=True, help="Pipeline name from configs/model_registry.yaml")
    parser.add_argument("--weights", help="Optional weights override")
    parser.add_argument("--data", help="Optional data override")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate, default val")
    parser.add_argument("--imgsz", type=int, help="Override image size")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--workers", type=int, help="Override workers")
    parser.add_argument("--conf", type=float, help="Detection confidence threshold")
    parser.add_argument("--device", default="cpu", help="Evaluation device, e.g. cpu, 0")
    parser.add_argument("--project", help="Output project override")
    parser.add_argument("--name", help="Run name override")
    return parser


if __name__ == "__main__":
    raise SystemExit(evaluate_pipeline(build_parser().parse_args()))
