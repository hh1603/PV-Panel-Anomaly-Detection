from __future__ import annotations

import argparse
import base64
import cgi
import html
import json
import re
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2

from common import PROJECT_ROOT, load_yaml, write_json
from detect_postprocess import (
    detect_summary,
    detection_postprocess_config,
    extract_detections,
    render_detection_image,
)
from run_by_modality import resolve_modality, resolve_pipeline_name
from run_pipeline import REGISTRY_PATH, classify_summary, import_yolo, load_pipeline_config, resolve_path

RUNS_ROOT = PROJECT_ROOT / "runs" / "web_demo"
UPLOAD_ROOT = RUNS_ROOT / "uploads"
RESULT_ROOT = RUNS_ROOT / "results"
MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass
class PredictionPayload:
    session_id: str
    created_at: str
    modality: str
    pipeline_name: str
    task: str
    summary: dict[str, Any]
    uploaded_path: Path
    original_image_path: Path
    result_image_path: Path
    summary_path: Path
    report_text_path: Path
    report_html_path: Path
    original_image_data_url: str
    result_image_base64: str


def slugify_filename(filename: str) -> str:
    source = Path(filename).name or "upload.jpg"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source).stem).strip("._") or "upload"
    suffix = Path(source).suffix.lower() or ".jpg"
    return f"{stem}{suffix}"


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)


def report_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_registry() -> dict[str, Any]:
    return load_yaml(REGISTRY_PATH)


def pipeline_overview() -> list[dict[str, str]]:
    registry = load_registry()
    routes = registry.get("modality_routes", {})
    pipelines = registry.get("pipelines", {})
    cards = []
    for modality in ("visible", "infrared"):
        route = routes.get(modality, {})
        pipeline_name = route.get("predict", "")
        pipeline = pipelines.get(pipeline_name, {})
        defaults = pipeline.get("default_predict", {})
        postprocess = pipeline.get("detect_postprocess", {}) if pipeline.get("task") == "detect" else {}
        weights = resolve_path(defaults.get("weights")) if defaults.get("weights") else None
        cards.append(
            {
                "modality": modality,
                "pipeline": pipeline_name or "-",
                "task": pipeline.get("task", "-"),
                "weights": str(weights) if weights else "-",
                "postprocess": json.dumps(postprocess, ensure_ascii=False) if postprocess else "-",
            }
        )
    return cards


def get_model(modality: str, device: str):
    normalized = resolve_modality(modality)
    pipeline_name = resolve_pipeline_name(normalized, "predict")
    pipeline = load_pipeline_config(pipeline_name)
    defaults = pipeline.get("default_predict", {})
    weights = resolve_path(defaults.get("weights"))
    if weights is None or not weights.exists():
        raise FileNotFoundError(f"Configured weights not found: {weights}")

    cache_key = (pipeline_name, str(weights), device)
    model = MODEL_CACHE.get(cache_key)
    if model is None:
        YOLO = import_yolo()
        model = YOLO(str(weights))
        MODEL_CACHE[cache_key] = model
    return normalized, pipeline_name, pipeline, model


def encode_result_image(image) -> str:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode result image.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".json": "application/json; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".html": "text/html; charset=utf-8",
    }
    return mapping.get(suffix, "application/octet-stream")


def data_url_for_path(path: Path) -> str:
    content_type = content_type_for_path(path).split(';', 1)[0]
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def detect_count(summary: dict[str, Any]) -> int:
    return len(summary.get("detections", []))


def classify_top1(summary: dict[str, Any]) -> dict[str, Any] | None:
    top1 = summary.get("top1")
    return top1 if isinstance(top1, dict) else None


def resolve_detection_profile(profile: str | None, pipeline: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    selected = (profile or "balanced").strip().lower()
    if selected == "sensitive":
        config = detection_postprocess_config(
            pipeline,
            argparse.Namespace(post_conf=0.03, min_area=0.003, max_area=0.03, max_detections=2),
        )
        return "sensitive", config
    return "balanced", detection_postprocess_config(pipeline, argparse.Namespace())


def report_conclusion(payload: PredictionPayload) -> str:
    if payload.task == "detect":
        count = detect_count(payload.summary)
        profile = str(payload.summary.get("profile") or "balanced")
        if count == 0:
            if profile == "balanced":
                return "在均衡过滤模式下未保留任何异常区域。可尝试切换到灵敏模式并人工复核图像。"
            return "在灵敏过滤模式下未保留任何异常区域。建议仍进行人工复核。"
        return f"检测到 {count} 个异常区域，请查看结果图中标注的位置。"

    top1 = classify_top1(payload.summary)
    if top1 is None:
        return "当前图像无可用分类结果。"
    return f"当前预测：{top1['class_name']}（置信度 {top1['confidence']:.4f}）。"


def render_kv_items(items: list[tuple[str, str]]) -> str:
    return "".join(
        f"<li><strong>{html.escape(label)}</strong>: {html.escape(value)}</li>" for label, value in items
    )


def report_highlights(payload: PredictionPayload) -> list[tuple[str, str]]:
    items = [
        ("生成时间", payload.created_at),
        ("会话ID", payload.session_id),
        ("图像模态", payload.modality),
        ("流水线", payload.pipeline_name),
        ("任务类型", payload.task),
        ("结论", report_conclusion(payload)),
    ]
    if payload.task == "detect":
        items.append(("检测模式", str(payload.summary.get("profile") or "balanced")))
        items.append(("检测数量", str(detect_count(payload.summary))))
    else:
        top1 = classify_top1(payload.summary)
        if top1 is not None:
            items.append(("Top1", f"{top1['class_name']} ({top1['confidence']:.4f})"))
    return items


def _conf_bar(conf: float) -> str:
    pct = min(100, int(conf * 100))
    color = "#f85149" if pct >= 50 else "#f0883e" if pct >= 15 else "#3fb950"
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px;">'
        f'<div style="flex:1;height:6px;background:#21262d;border-radius:3px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:3px;"></div></div>'
        f'<span style="font-size:12px;color:#e6edf3;min-width:44px;">{conf:.4f}</span></div>'
    )


def render_result_details(payload: PredictionPayload) -> str:
    if payload.task == "detect":
        detections = payload.summary.get("detections", [])
        if not detections:
            return '<li style="color:#7d8590;">未检测到异常框（已通过后处理过滤）</li>'
        items = []
        for i, item in enumerate(detections, 1):
            bar = _conf_bar(item['confidence'])
            items.append(
                f'<li><div style="display:flex;justify-content:space-between;">'
                f'<strong style="color:#f0883e;">#{i} {html.escape(item["class_name"])}</strong>'
                f'<span style="font-size:12px;color:#7d8590;">面积占比 {format_percent(item.get("area_ratio"))}</span></div>'
                f'{bar}'
                f'<div style="font-size:11px;color:#484f58;margin-top:3px;">框坐标: {html.escape(str(item["xyxy"]))}</div></li>'
            )
        return "".join(items)

    predictions = payload.summary.get("predictions", [])
    if not predictions:
        return '<li style="color:#7d8590;">没有可展示的分类结果。</li>'
    items = []
    for i, item in enumerate(predictions, 1):
        bar = _conf_bar(item['confidence'])
        items.append(
            f'<li><strong style="color:#3b9eff;">#{i} {html.escape(item["class_name"])}</strong>{bar}</li>'
        )
    return "".join(items)


def build_report_text(payload: PredictionPayload) -> str:
    lines = [
        "光伏板巡检报告",
        f"生成时间: {payload.created_at}",
        f"会话ID: {payload.session_id}",
        f"图像模态: {payload.modality}",
        f"流水线: {payload.pipeline_name}",
        f"任务类型: {payload.task}",
        f"检测模式: {payload.summary.get('profile', '-')}",
        f"上传图像: {payload.uploaded_path}",
        f"结果图像: {payload.result_image_path}",
        f"摘要JSON: {payload.summary_path}",
        f"可打印报告: {payload.report_html_path}",
        f"结论: {report_conclusion(payload)}",
        "",
        "检测详情:",
    ]

    if payload.task == "detect":
        detections = payload.summary.get("detections", [])
        if not detections:
            lines.append("1. 未保留任何异常框。")
        else:
            for index, item in enumerate(detections, start=1):
                lines.append(
                    f"{index}. {item['class_name']} | confidence={item['confidence']:.4f} | area_ratio={format_percent(item.get('area_ratio'))} | box={item['xyxy']}"
                )
    else:
        predictions = payload.summary.get("predictions", [])
        if not predictions:
            lines.append("1. 无分类结果。")
        else:
            for index, item in enumerate(predictions, start=1):
                lines.append(f"{index}. {item['class_name']} | confidence={item['confidence']:.4f}")

    return "\n".join(lines) + "\n"


def render_print_report(payload: PredictionPayload) -> str:
    highlights_html = render_kv_items(report_highlights(payload))
    details_html = render_result_details(payload)
    raw_json = html.escape(json.dumps(payload.summary, ensure_ascii=False, indent=2))
    report_download_url = f"/download/{payload.session_id}/report.html"
    text_download_url = f"/download/{payload.session_id}/report.txt"
    summary_download_url = f"/download/{payload.session_id}/summary.json"
    original_view_url = f"/artifact/{payload.session_id}/original"
    result_view_url = f"/artifact/{payload.session_id}/result"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>光伏板巡检报告</title>
  <style>
    :root {{
      --ink: #18352f;
      --muted: #52635e;
      --line: #d8ddd9;
      --accent: #1f7a6d;
      --paper: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Microsoft YaHei", "Segoe UI", sans-serif; color: var(--ink); background: #f5f6f4; }}
    .page {{ width: min(1120px, calc(100% - 32px)); margin: 24px auto 40px; background: var(--paper); border: 1px solid var(--line); border-radius: 20px; padding: 28px; box-shadow: 0 12px 34px rgba(24, 53, 47, 0.08); }}
    .toolbar {{ display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }}
    .button {{ display: inline-flex; align-items: center; justify-content: center; padding: 12px 16px; border-radius: 14px; border: 1px solid var(--line); text-decoration: none; color: var(--ink); background: #fff; font-weight: 700; cursor: pointer; }}
    .button.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .small-button {{ padding: 9px 12px; font-size: 13px; }}
    .eyebrow {{ margin: 0 0 10px; color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: 0.18em; font-weight: 700; }}
    h1, h2 {{ margin: 0; font-family: "Georgia", "Times New Roman", serif; }}
    h1 {{ font-size: 34px; margin-bottom: 10px; }}
    h2 {{ font-size: 22px; margin-bottom: 10px; }}
    p {{ line-height: 1.7; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid var(--line); border-radius: 18px; padding: 18px; background: #fff; }}
    .image-card {{ display: grid; gap: 12px; }}
    .image-actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .preview-image {{ width: 100%; height: auto; display: block; border-radius: 18px; border: 1px solid var(--line); background: #fff; object-fit: contain; }}
    .list {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }}
    .list li {{ padding: 12px 14px; border: 1px solid var(--line); border-radius: 14px; background: #fcfcfb; }}
    .file-links {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }}
    pre {{ margin: 0; padding: 16px; background: #102723; color: #edf6f3; border-radius: 16px; overflow: auto; white-space: pre-wrap; word-break: break-word; font-size: 13px; }}
    .small {{ font-size: 13px; word-break: break-all; }}
    @media print {{
      body {{ background: #fff; }}
      .page {{ width: 100%; margin: 0; border: 0; border-radius: 0; box-shadow: none; padding: 0; }}
      .toolbar, .image-actions, .file-links {{ display: none; }}
      a {{ color: inherit; text-decoration: none; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <div class="toolbar">
      <button class="button primary" onclick="window.print()">打印报告</button>
      <a class="button" href="{report_download_url}">下载 HTML 报告</a>
      <a class="button" href="{text_download_url}">下载 TXT 报告</a>
      <a class="button" href="{summary_download_url}">下载 JSON 结果</a>
      <a class="button" href="/">返回上传页</a>
    </div>
    <p class="eyebrow">可打印报告</p>
    <h1>光伏板巡检报告</h1>
    <p>{html.escape(report_conclusion(payload))}</p>
    <section class="grid" style="margin-bottom: 18px;">
      <section class="card image-card">
        <div>
          <p class="eyebrow">原始图像</p>
          <div class="image-actions">
            <a class="button small-button" href="{original_view_url}" target="_blank" rel="noopener">查看原图</a>
          </div>
        </div>
        <img class="preview-image" src="{payload.original_image_data_url}" alt="原始图像">
      </section>
      <section class="card image-card">
        <div>
          <p class="eyebrow">结果图像</p>
          <div class="image-actions">
            <a class="button small-button" href="{result_view_url}" target="_blank" rel="noopener">查看结果</a>
          </div>
        </div>
        <img class="preview-image" src="data:image/jpeg;base64,{payload.result_image_base64}" alt="结果图像">
      </section>
    </section>
    <section class="grid">
      <section class="card">
        <p class="eyebrow">摘要</p>
        <ul class="list">{highlights_html}</ul>
      </section>
      <section class="card">
        <p class="eyebrow">文件与链接</p>
        <ul class="list">
          <li>
            <strong>原始图像</strong>
            <div class="small">{html.escape(str(payload.original_image_path))}</div>
            <div class="file-links"><a class="button small-button" href="{original_view_url}" target="_blank" rel="noopener">查看原图</a></div>
          </li>
          <li>
            <strong>结果图像</strong>
            <div class="small">{html.escape(str(payload.result_image_path))}</div>
            <div class="file-links"><a class="button small-button" href="{result_view_url}" target="_blank" rel="noopener">查看结果</a></div>
          </li>
          <li>
            <strong>摘要 JSON</strong>
            <div class="small">{html.escape(str(payload.summary_path))}</div>
            <div class="file-links"><a class="button small-button" href="{summary_download_url}">下载 JSON</a></div>
          </li>
          <li>
            <strong>HTML 报告</strong>
            <div class="small">{html.escape(str(payload.report_html_path))}</div>
            <div class="file-links"><a class="button small-button" href="{report_download_url}">下载 HTML</a></div>
          </li>
          <li>
            <strong>TXT 报告</strong>
            <div class="small">{html.escape(str(payload.report_text_path))}</div>
            <div class="file-links"><a class="button small-button" href="{text_download_url}">下载 TXT</a></div>
          </li>
        </ul>
      </section>
    </section>
    <section class="card" style="margin-top: 18px;">
      <p class="eyebrow">检测详情</p>
      <ul class="list">{details_html}</ul>
    </section>
    <section class="card" style="margin-top: 18px;">
      <p class="eyebrow">原始 JSON</p>
      <pre>{raw_json}</pre>
    </section>
  </main>
</body>
</html>
"""


def write_prediction_reports(payload: PredictionPayload) -> None:
    payload.report_text_path.write_text(build_report_text(payload), encoding="utf-8")
    payload.report_html_path.write_text(render_print_report(payload), encoding="utf-8")


def predict_file(modality: str, image_path: Path, device: str = "cpu", detection_profile: str = "balanced") -> PredictionPayload:
    normalized, pipeline_name, pipeline, model = get_model(modality, device)
    defaults = pipeline.get("default_predict", {})
    task = pipeline["task"]

    predict_kwargs = {
        "source": str(image_path),
        "imgsz": defaults.get("imgsz", 640 if task == "detect" else 224),
        "device": device,
        "verbose": False,
    }
    if task == "detect":
        predict_kwargs["conf"] = defaults.get("conf", 0.25)

    result = model.predict(**predict_kwargs)[0]

    if task == "detect":
        selected_profile, postprocess = resolve_detection_profile(detection_profile, pipeline)
        detections = extract_detections(result, image_path, postprocess)
        summary = detect_summary(image_path, detections, postprocess)
        summary["profile"] = selected_profile
        rendered = render_detection_image(image_path, detections)
    elif task == "classify":
        summary = classify_summary(result, image_path, int(defaults.get("topk", 3)))
        rendered = result.plot()
    else:
        raise ValueError(f"Unsupported task: {task}")

    session_id = timestamp_token()
    created_at = report_timestamp()
    session_dir = RESULT_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    original_image_path = session_dir / f"{image_path.stem}_original{image_path.suffix.lower()}"
    result_image_path = session_dir / f"{image_path.stem}_{normalized}_result.jpg"
    summary_path = session_dir / f"{image_path.stem}_{normalized}_summary.json"
    report_text_path = session_dir / "report.txt"
    report_html_path = session_dir / "report.html"
    shutil.copy2(image_path, original_image_path)
    cv2.imwrite(str(result_image_path), rendered)
    write_json(summary_path, summary)

    payload = PredictionPayload(
        session_id=session_id,
        created_at=created_at,
        modality=normalized,
        pipeline_name=pipeline_name,
        task=task,
        summary=summary,
        uploaded_path=image_path,
        original_image_path=original_image_path,
        result_image_path=result_image_path,
        summary_path=summary_path,
        report_text_path=report_text_path,
        report_html_path=report_html_path,
        original_image_data_url=data_url_for_path(original_image_path),
        result_image_base64=encode_result_image(rendered),
    )
    write_prediction_reports(payload)
    return payload


def save_upload(filename: str, data: bytes) -> Path:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    token = timestamp_token()
    safe_name = slugify_filename(filename)
    target = UPLOAD_ROOT / f"{token}_{safe_name}"
    target.write_bytes(data)
    return target


def render_layout(title: str, body: str) -> bytes:
    cards = "".join(
        f"""
        <article class=\"pipeline-card\">
          <p class=\"section-label section-label-green\">{html.escape(item['modality'])}</p>
          <h3>{html.escape(item['pipeline'])}</h3>
          <p><span class=\"tag\">{html.escape(item['task'])}</span></p>
          <p class=\"path\">{html.escape(item['weights'])}</p>
        </article>
        """
        for item in pipeline_overview()
    )

    page = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(title)} — 光伏异常检测</title>
  <style>
    :root {{
      --bg:#0d1117;--surface:#161b22;--surface2:#21262d;
      --border:rgba(255,255,255,0.08);--border2:rgba(255,255,255,0.13);
      --ink:#e6edf3;--muted:#7d8590;--faint:#484f58;
      --accent:#3b9eff;--accent-dim:rgba(59,158,255,0.12);
      --green:#3fb950;--green-dim:rgba(63,185,80,0.12);
      --orange:#f0883e;--danger:#f85149;
      --r:10px;--shadow:0 8px 32px rgba(0,0,0,0.45);
    }}
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
    body{{
      min-height:100vh;color:var(--ink);background:var(--bg);
      font-family:-apple-system,"PingFang SC","Microsoft YaHei","Segoe UI",sans-serif;
      font-size:15px;line-height:1.65;
      background-image:radial-gradient(ellipse 80% 35% at 50% -5%,rgba(59,158,255,0.07),transparent);
    }}
    a{{color:var(--accent);text-decoration:none;}}
    a:hover{{text-decoration:underline;}}
    nav{{
      position:sticky;top:0;z-index:100;
      height:52px;padding:0 20px;
      display:flex;align-items:center;gap:10px;
      background:rgba(13,17,23,0.88);backdrop-filter:blur(14px);
      border-bottom:1px solid var(--border);
    }}
    .nav-logo{{
      width:28px;height:28px;border-radius:7px;flex-shrink:0;
      background:linear-gradient(135deg,#3b9eff,#3fb950);
      display:flex;align-items:center;justify-content:center;font-size:15px;
    }}
    .nav-title{{font-weight:700;font-size:14px;color:var(--ink);}}
    .nav-pill{{
      padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;
      background:var(--green-dim);color:var(--green);border:1px solid rgba(63,185,80,0.25);
    }}
    .nav-pill::before{{content:"\u25cf ";font-size:8px;}}
    .nav-space{{flex:1;}}
    .nav-hint{{font-size:12px;color:var(--muted);}}
    .shell{{width:min(1160px,calc(100% - 32px));margin:28px auto 72px;display:grid;gap:20px;}}
    .hero{{
      position:relative;overflow:hidden;
      background:linear-gradient(135deg,var(--surface) 0%,#0b1628 100%);
      border:1px solid var(--border2);border-radius:14px;
      padding:36px 32px;box-shadow:var(--shadow);
    }}
    .hero::before{{
      content:"";position:absolute;inset:0;
      background:radial-gradient(circle at 90% 10%,rgba(59,158,255,0.08),transparent 55%);
      pointer-events:none;
    }}
    .status-badge{{
      display:inline-flex;align-items:center;gap:6px;
      padding:3px 11px;border-radius:20px;font-size:12px;font-weight:600;
      background:var(--green-dim);color:var(--green);border:1px solid rgba(63,185,80,0.28);
      margin-bottom:16px;
    }}
    .status-dot{{width:7px;height:7px;border-radius:50%;background:var(--green);flex-shrink:0;box-shadow:0 0 6px var(--green);}}
    .hero h1{{font-size:clamp(22px,3.5vw,34px);font-weight:700;color:var(--ink);margin-bottom:10px;line-height:1.2;}}
    .hero p{{color:var(--muted);font-size:14px;max-width:520px;}}
    .two-col{{display:grid;grid-template-columns:1fr 320px;gap:20px;align-items:start;}}
    @media(max-width:860px){{.two-col{{grid-template-columns:1fr;}}}}
    .card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:22px;box-shadow:var(--shadow);}}
    .section-label{{font-size:11px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);margin-bottom:10px;display:block;}}
    .section-label-green{{color:var(--green);}}
    .section-label-orange{{color:var(--orange);}}
    .form-grid{{display:grid;gap:16px;}}
    .field{{display:grid;gap:6px;}}
    .field label{{font-size:13px;font-weight:600;color:var(--muted);}}
    select,input[type=file]{{
      width:100%;padding:10px 14px;
      background:var(--surface2);color:var(--ink);
      border:1px solid var(--border2);border-radius:8px;
      font:inherit;font-size:14px;transition:border-color .15s;
    }}
    select:focus,input[type=file]:focus{{outline:none;border-color:var(--accent);}}
    .btn{{
      display:inline-flex;align-items:center;justify-content:center;gap:7px;
      padding:10px 18px;border-radius:8px;border:none;
      font:inherit;font-size:14px;font-weight:600;
      cursor:pointer;transition:opacity .15s,transform .12s;text-decoration:none;
    }}
    .btn:hover{{opacity:.85;transform:translateY(-1px);text-decoration:none;}}
    .btn-primary{{background:var(--accent);color:#fff;width:100%;padding:12px;}}
    .btn-ghost{{background:var(--surface2);color:var(--ink);border:1px solid var(--border2);}}
    .tips-list{{list-style:none;display:grid;gap:7px;margin-top:14px;}}
    .tips-list li{{
      font-size:13px;color:var(--muted);padding:9px 13px;
      background:var(--surface2);border-radius:8px;border-left:3px solid var(--border2);
    }}
    .pipeline-card{{padding:14px;border-radius:9px;background:var(--surface2);border:1px solid var(--border);margin-bottom:10px;}}
    .pipeline-card h3{{font-size:13px;font-weight:700;color:var(--ink);margin-bottom:5px;}}
    .tag{{display:inline-block;padding:2px 8px;border-radius:6px;font-size:11px;font-weight:600;background:var(--accent-dim);color:var(--accent);border:1px solid rgba(59,158,255,.2);}}
    .path{{word-break:break-all;font-size:11px;color:var(--faint);margin-top:5px;}}
    .result-image{{width:100%;display:block;border-radius:10px;border:1px solid var(--border);margin:16px 0;}}
    .kv-list{{list-style:none;display:grid;gap:8px;}}
    .kv-list li{{
      display:flex;justify-content:space-between;align-items:baseline;
      padding:9px 13px;background:var(--surface2);border-radius:8px;
      border:1px solid var(--border);font-size:13px;gap:10px;
    }}
    .kv-list li strong{{color:var(--muted);font-weight:600;flex-shrink:0;}}
    .det-list{{list-style:none;display:grid;gap:8px;}}
    .det-list li{{
      padding:10px 13px;background:var(--surface2);
      border-radius:8px;border:1px solid var(--border);
      font-size:13px;color:var(--ink);border-left:3px solid var(--orange);
    }}
    .action-row{{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0;}}
    .report-card{{margin:16px 0;padding:18px;border-radius:10px;background:var(--surface2);border:1px solid var(--border);}}
    .two-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;}}
    .compare-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0;}}
    .compare-item{{display:grid;gap:8px;}}
    @media(max-width:600px){{.compare-grid{{grid-template-columns:1fr;}}}}
    @media(max-width:600px){{.two-grid{{grid-template-columns:1fr;}}}}
    pre{{margin:0;white-space:pre-wrap;word-break:break-word;background:#0d1f1a;color:#adbfb8;border-radius:10px;padding:16px;overflow:auto;font-size:12px;border:1px solid var(--border);}}
    .error{{padding:12px 14px;border-radius:8px;background:rgba(248,81,73,0.1);border:1px solid rgba(248,81,73,.3);color:#f85149;font-size:14px;margin-bottom:12px;}}
    .metric-list,.detection-list{{list-style:none;display:grid;gap:8px;}}
    .metric-list li,.detection-list li{{padding:9px 13px;background:var(--surface2);border-radius:8px;border:1px solid var(--border);font-size:13px;color:var(--ink);}}
    .no-print{{}}
    @media(max-width:720px){{
      .shell{{width:calc(100% - 24px);}}
      .hero{{padding:22px 18px;}}
      .btn{{width:100%;}}
      .action-row{{flex-direction:column;}}
    }}
    @media print{{
      body{{background:#fff;color:#000;}}
      nav,.hero,.action-row.no-print,.raw-json.no-print{{display:none!important;}}
      .card{{box-shadow:none;border:1px solid #ddd;background:#fff;}}
      .two-col{{grid-template-columns:1fr;}}
    }}
  </style>
</head>
<body>
  <nav>
    <div class=\"nav-logo\">\u2600</div>
    <span class=\"nav-title\">光伏异常检测系统</span>
    <span class=\"nav-pill\">运行中</span>
    <div class=\"nav-space\"></div>
    <span class=\"nav-hint\">本地推理 · CPU</span>
  </nav>
  <main class=\"shell\">
    <section class=\"hero no-print\">
      <div class=\"status-badge\"><span class=\"status-dot\"></span>系统就绪</div>
      <h1>光伏板异常检测<br>可视化平台</h1>
      <p>上传可见光或红外图像，系统自动选择对应模型进行推理，结果实时展示并可导出报告。</p>
    </section>
    <div class=\"two-col\">
      <section class=\"card\">{body}</section>
      <aside class=\"card no-print\">
        <span class=\"section-label section-label-green\">当前激活流水线</span>
        <div>{cards}</div>
      </aside>
    </div>
  </main>
</body>
</html>
"""
    return page.encode('utf-8')


def render_home_page(message: str | None = None) -> bytes:
    notice = f'<p class="error">{html.escape(message)}</p>' if message else ""
    body = f"""
      <p class="eyebrow">上传图像</p>
      <h2>单张图像检测</h2>
      <p>当前原型中，<code>visible/rgb</code> 使用检测器，<code>infrared/ir</code> 使用分类器。若可见光缺陷较隐蔽，可将检测模式切换为<strong>灵敏</strong>模式。</p>
      {notice}
      <form action="/predict" method="post" enctype="multipart/form-data">
        <label for="modality">图像模态
          <select id="modality" name="modality" required>
            <option value="visible">可见光 / rgb</option>
            <option value="infrared">红外 / ir</option>
          </select>
        </label>
        <label for="profile">检测模式
          <select id="profile" name="profile">
            <option value="balanced">均衡（默认）</option>
            <option value="sensitive">灵敏 / 高召回</option>
          </select>
        </label>
        <div class="field">
          <label for="image">上传图像</label>
          <div id="drop-zone" onclick="document.getElementById('image').click()" ondragover="event.preventDefault();this.classList.add('drag-over')" ondragleave="this.classList.remove('drag-over')" ondrop="handleDrop(event)">
            <div id="drop-hint">&#128194; 点击选择或拖拽图片到此处</div>
            <img id="preview-img" style="display:none;max-height:180px;border-radius:8px;margin-top:8px;" alt="预览">
          </div>
          <input id="image" name="image" type="file" accept="image/*" required style="display:none" onchange="if(this.files[0])showPreview(this.files[0])">
        </div>
        <button type="submit" id="submit-btn">开始检测</button>
      </form>
      <style>
      #drop-zone{border:2px dashed var(--border2);border-radius:8px;padding:20px;text-align:center;cursor:pointer;transition:border-color .15s,background .15s;background:var(--surface2);}
      #drop-zone:hover,#drop-zone.drag-over{border-color:var(--accent);background:var(--accent-dim);}
      #drop-hint{color:var(--muted);font-size:14px;}
      </style>
      <ul class="tips-list">
        <li>请上传单张 JPG 或 PNG 图像。</li>
        <li><strong>均衡</strong>模式过滤大范围误检框，适合演示。</li>
        <li><strong>灵敏</strong>模式保留更弱的检测，适合边缘样本复查。</li>
      </ul>
    """
    return render_layout("光伏板异常检测演示", body)


def render_result_page(payload: PredictionPayload) -> bytes:
    highlights_html = render_kv_items(report_highlights(payload))
    details_html = render_result_details(payload)
    raw_json = html.escape(json.dumps(payload.summary, ensure_ascii=False, indent=2))
    report_url = f"/report/{payload.session_id}"
    report_download_url = f"/download/{payload.session_id}/report.html"
    text_download_url = f"/download/{payload.session_id}/report.txt"

    body = f"""
      <span class="section-label section-label-orange">检测完成</span>
      <h2 style="margin-bottom:6px;">检测结果</h2>
      <p style="margin-bottom:14px;">流水线：<strong>{html.escape(payload.pipeline_name)}</strong> · 任务：<strong>{html.escape(payload.task)}</strong></p>
      <div class="action-row no-print">
        <a class="btn btn-ghost" href="/">← 返回上传</a>
        <a class="btn btn-ghost" href="{report_url}" target="_blank" rel="noopener">打开报告</a>
        <a class="btn btn-ghost" href="{report_download_url}">下载 HTML</a>
        <a class="btn btn-ghost" href="{text_download_url}">下载 TXT</a>
        <button class="btn btn-ghost" type="button" onclick="window.print()">打印</button>
      </div>
      <div class="compare-grid">
        <div class="compare-item">
          <p class="section-label" style="margin-bottom:8px;">原始图像</p>
          <img class="result-image" src="{payload.original_image_data_url}" alt="原始图像">
        </div>
        <div class="compare-item">
          <p class="section-label section-label-orange" style="margin-bottom:8px;">检测结果</p>
          <img class="result-image" src="data:image/jpeg;base64,{payload.result_image_base64}" alt="检测结果">
        </div>
      </div>
      <div class="two-grid">
        <section>
          <p class="section-label" style="margin-bottom:8px;">检测摘要</p>
          <ul class="kv-list">{highlights_html}</ul>
        </section>
        <section>
          <p class="section-label section-label-orange" style="margin-bottom:8px;">检测框详情</p>
          <ul class="det-list">{details_html}</ul>
        </section>
      </div>
      <section class="raw-json no-print" style="margin-top:16px;">
        <p class="section-label" style="margin-bottom:8px;">原始 JSON</p>
        <pre>{raw_json}</pre>
      </section>
    """
    return render_layout("检测结果", body)


def safe_session_report_path(session_id: str) -> Path | None:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        return None
    report_path = RESULT_ROOT / session_id / "report.html"
    return report_path if report_path.exists() else None


def safe_session_file(session_id: str, pattern: str) -> Path | None:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        return None
    session_dir = RESULT_ROOT / session_id
    if not session_dir.exists():
        return None
    matches = sorted(session_dir.glob(pattern))
    return matches[0] if matches else None


def safe_session_asset(session_id: str, asset_name: str) -> tuple[Path, str] | None:
    patterns = {
        "original": "*_original.*",
        "result": "*_result.*",
    }
    pattern = patterns.get(asset_name)
    if pattern is None:
        return None
    file_path = safe_session_file(session_id, pattern)
    if file_path is None:
        return None
    return file_path, content_type_for_path(file_path)


def safe_session_download(session_id: str, filename: str) -> tuple[Path, str, str] | None:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        return None

    session_dir = RESULT_ROOT / session_id
    if not session_dir.exists():
        return None

    if filename == "report.html":
        path = session_dir / "report.html"
        if not path.exists():
            return None
        return path, "text/html; charset=utf-8", f"solar_panel_report_{session_id}.html"

    if filename == "report.txt":
        path = session_dir / "report.txt"
        if not path.exists():
            return None
        return path, "text/plain; charset=utf-8", f"solar_panel_report_{session_id}.txt"

    if filename == "summary.json":
        path = safe_session_file(session_id, "*_summary.json")
        if path is None:
            return None
        return path, "application/json; charset=utf-8", f"solar_panel_report_{session_id}_summary.json"

    return None


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "SolarPanelDemo/1.1"

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/health":
            self.respond(HTTPStatus.OK, b'{"status":"ok"}', "application/json; charset=utf-8")
            return
        if route == "/":
            self.respond(HTTPStatus.OK, render_home_page(), "text/html; charset=utf-8")
            return
        if route.startswith("/report/"):
            session_id = route.removeprefix("/report/").strip()
            report_path = safe_session_report_path(session_id)
            if report_path is None:
                self.respond(HTTPStatus.NOT_FOUND, render_home_page("??????????????????"), "text/html; charset=utf-8")
                return
            self.respond(HTTPStatus.OK, report_path.read_bytes(), "text/html; charset=utf-8")
            return
        if route.startswith("/artifact/"):
            parts = [part for part in route.split("/") if part]
            if len(parts) != 3:
                self.respond(HTTPStatus.NOT_FOUND, render_home_page("????????"), "text/html; charset=utf-8")
                return
            _, session_id, asset_name = parts
            asset_payload = safe_session_asset(session_id, asset_name)
            if asset_payload is None:
                self.respond(HTTPStatus.NOT_FOUND, render_home_page("????????????????????"), "text/html; charset=utf-8")
                return
            file_path, content_type = asset_payload
            self.respond(HTTPStatus.OK, file_path.read_bytes(), content_type)
            return
        if route.startswith("/download/"):
            parts = [part for part in route.split("/") if part]
            if len(parts) != 3:
                self.respond(HTTPStatus.NOT_FOUND, render_home_page("????????"), "text/html; charset=utf-8")
                return
            _, session_id, filename = parts
            download_payload = safe_session_download(session_id, filename)
            if download_payload is None:
                self.respond(HTTPStatus.NOT_FOUND, render_home_page("????????????????????"), "text/html; charset=utf-8")
                return
            file_path, content_type, download_name = download_payload
            self.respond(
                HTTPStatus.OK,
                file_path.read_bytes(),
                content_type,
                {"Content-Disposition": f'attachment; filename="{download_name}"'},
            )
            return
        self.respond(HTTPStatus.NOT_FOUND, render_home_page("????????????????"), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route != "/predict":
            self.respond(HTTPStatus.NOT_FOUND, render_home_page("提交地址不存在。"), "text/html; charset=utf-8")
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.respond(HTTPStatus.BAD_REQUEST, render_home_page("请求格式无效，请重新上传图像。"), "text/html; charset=utf-8")
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
        )

        modality = form.getfirst("modality", "").strip()
        profile = form.getfirst("profile", "balanced").strip().lower()
        file_item = form["image"] if "image" in form else None
        if not modality:
            self.respond(HTTPStatus.BAD_REQUEST, render_home_page("请选择图像模态。"), "text/html; charset=utf-8")
            return
        if profile not in {"balanced", "sensitive"}:
            profile = "balanced"
        if file_item is None or not getattr(file_item, "file", None):
            self.respond(HTTPStatus.BAD_REQUEST, render_home_page("未收到图像文件，请重新上传。"), "text/html; charset=utf-8")
            return

        raw = file_item.file.read()
        if not raw:
            self.respond(HTTPStatus.BAD_REQUEST, render_home_page("上传的文件为空，请重新选择图像。"), "text/html; charset=utf-8")
            return

        filename = file_item.filename or "upload.jpg"
        try:
            uploaded_path = save_upload(filename, raw)
            payload = predict_file(modality, uploaded_path, device=self.server.device, detection_profile=profile)  # type: ignore[attr-defined]
            self.respond(HTTPStatus.OK, render_result_page(payload), "text/html; charset=utf-8")
        except Exception as exc:
            self.respond(HTTPStatus.INTERNAL_SERVER_ERROR, render_home_page(f"检测失败：{exc}"), "text/html; charset=utf-8")

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def respond(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


class DemoServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class, device: str):
        super().__init__(server_address, handler_class)
        self.device = device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a local web demo for solar panel inference.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind, default 127.0.0.1")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind, default 7860")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, 0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    server = DemoServer((args.host, args.port), DemoHandler, args.device)
    print(f"Demo server running at http://{args.host}:{args.port} (device={args.device})")
    print("Use Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down demo server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
