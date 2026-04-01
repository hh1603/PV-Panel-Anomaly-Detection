# Solar Panel Fault Detection Baseline

This workspace now contains practical baselines aligned with the current data situation:

- `visible_detector_boosted`: recommended visible-light defect detector using a boosted train split and tuned demo-time postprocess
- `visible_detector`: original visible-light detector baseline kept for comparison
- `infrared_classifier`: infrared hotspot severity classification with YOLO-CLS
- optional SAM refinement after detection
- a registry-driven runner in `scripts/run_pipeline.py`
- a modality-driven runner in `scripts/run_by_modality.py`
- a shared detection postprocess module in `scripts/detect_postprocess.py`
- a local web demo in `scripts/web_demo.py`
- a Windows launcher in `scripts/launch_demo.cmd`
- a unified evaluation script in `scripts/evaluate_pipelines.py`
- detection dataset analysis and boost scripts for the visible detector
- local `.venv` and `.ultralytics` isolation inside the project

## Current data conclusion

- `dataset_1/` supports detection, but only as a single-class `defect` task
- `train/`, `val/`, `test/` support visible-light classification
- `LLM-PV-image-database/*.zip` has visible / IR / EL classification resources
- there is still no infrared detection box dataset in the workspace

This means the project currently supports:

- visible-light detection
- infrared classification

It does **not** yet support a true infrared detector with fault boxes.

## Environment

Create the local environment once:

```cmd
.\scripts\setup_venv.cmd
```

Run everything with the project interpreter:

```powershell
.\.venv\Scripts\python.exe <script>
```

## Data preparation

Convert `dataset_1` into YOLO detection format:

```powershell
.\.venv\Scripts\python.exe .\scripts\convert_dataset_1_to_yolo.py
```

Extract visible / IR / EL archive datasets:

```powershell
.\.venv\Scripts\python.exe .\scripts\extract_llm_pv_archives.py --overwrite
```

Build ready-to-train classification splits:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_llm_pv_cls_dataset.py --overwrite
```

## Simplest visual demo

Start the local upload page:

```cmd
.\scripts\launch_demo.cmd
```

Or run the Python script directly:

```powershell
.\.venv\Scripts\python.exe .\scripts\web_demo.py --host 127.0.0.1 --port 7860 --device cpu
```

Then open:

```text
http://127.0.0.1:7860
```

What the demo does now:

- if you choose `visible` or `rgb`, it routes to `visible_detector_boosted`
- if you choose `infrared` or `ir`, it routes to `infrared_classifier`
- visible detection now uses a tuned postprocess stage to suppress oversized false-positive boxes
- uploaded files are stored in `runs/web_demo/uploads`
- rendered results and JSON summaries are stored in `runs/web_demo/results`

## Named pipelines

The central registry is `configs/model_registry.yaml`.

Current modality defaults:

- `visible -> visible_detector_boosted`
- `infrared -> infrared_classifier`

Important visible pipelines:

- `visible_detector_boosted`: promoted visible detector with default predict weight `runs/train/visible_detector_boosted_baseline_cpu/weights/best.pt`
- `visible_detector`: original visible baseline kept only for comparison

## Simplest unified entry

If you do not want to remember pipeline names, use the modality router:

Visible-light inference with the default demo preset:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_by_modality.py predict --modality visible --source .\datasets\dataset_1_defect\images\val\039R.jpg --device cpu
```

Infrared inference:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_by_modality.py predict --modality infrared --source .\datasets\classification_ready\ir_multiclass\test\healthy\22_healthy.jpg --device cpu
```

The same script also accepts `rgb` as an alias for `visible`, and `ir` as an alias for `infrared`.

## Visible postprocess presets

### Demo-first preset

This is the current default for `visible_detector_boosted`:

- base YOLO predict threshold: `0.001`
- postprocess confidence: `0.15`
- min box area ratio: `0.003`
- max box area ratio: `0.03`
- max detections kept: `1`

This preset is designed to keep the visible-light demo cleaner by filtering out very large false-positive boxes.

### Recall-first override

If you want a looser visible setting for debugging or case review, override the postprocess from CLI:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_by_modality.py predict --modality visible --source .\datasets\dataset_1_defect\images\test\083R.jpg --device cpu --post-conf 0.03 --min-area 0.003 --max-area 0.03 --max-detections 2
```

Current verified example:

- demo preset keeps a clean detection on `039R.jpg` and suppresses detections on a negative sample like `013R.jpg`
- recall override can recover detections on harder positive cases such as `083R.jpg`, but may introduce more false positives on negatives

## Boost the visible detector dataset

Create the boosted visible detection dataset by repeating only positive training images:

```powershell
.\.venv\Scripts\python.exe .\scripts\boost_detection_dataset.py --positive-repeat 4
```

This creates:

- `datasets/dataset_1_defect_boosted`
- `configs/dataset_1_defect_boosted.yaml`

Current boosted train statistics:

- original train split: `95` images, `18` positive images, positive ratio `0.189474`
- boosted train split: `167` images, `90` positive images, positive ratio `0.538922`
- validation and test splits remain unchanged

## Analyze the visible detection dataset

Run the detection dataset analyzer:

```powershell
.\.venv\Scripts\python.exe .\scripts\analyze_detection_dataset.py --data .\configs\dataset_1_defect.yaml --name dataset_1_defect_analysis
```

Run the boosted dataset analyzer:

```powershell
.\.venv\Scripts\python.exe .\scripts\analyze_detection_dataset.py --data .\configs\dataset_1_defect_boosted.yaml --name dataset_1_defect_boosted_analysis
```

Key finding from the original visible detection dataset:

- positive ratio stays around `18%` to `21%`
- most defect boxes fall into the `0.5% to 2%` image-area bucket
- the visible detector is therefore hit by both class imbalance and small-object difficulty

## Train with the unified runner

Recommended visible detector:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_pipeline.py train --pipeline visible_detector_boosted --device cpu --name visible_detector_boosted_baseline_cpu
```

Original visible baseline:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_pipeline.py train --pipeline visible_detector --device cpu
```

Infrared classifier:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_pipeline.py train --pipeline infrared_classifier --device cpu
```

If you have a GPU, replace `--device cpu` with `--device 0` and optionally raise batch size / image size.

## Evaluate current baselines

Original visible baseline on validation:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_pipelines.py --pipeline visible_detector --device cpu --imgsz 256 --batch 1 --workers 0 --name visible_detector_eval_baseline
```

Original visible baseline on test:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_pipelines.py --pipeline visible_detector --device cpu --imgsz 256 --batch 1 --workers 0 --split test --name visible_detector_eval_test_baseline
```

Boosted visible detector on validation:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_pipelines.py --pipeline visible_detector_boosted --weights .\runs\train\visible_detector_boosted_baseline_cpu\weights\best.pt --device cpu --imgsz 320 --batch 1 --workers 0 --name visible_detector_boosted_eval_val
```

Boosted visible detector on test:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_pipelines.py --pipeline visible_detector_boosted --weights .\runs\train\visible_detector_boosted_baseline_cpu\weights\best.pt --device cpu --imgsz 320 --batch 1 --workers 0 --split test --name visible_detector_boosted_eval_test
```

Infrared classifier evaluation:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_pipelines.py --pipeline infrared_classifier --device cpu --imgsz 224 --batch 16 --workers 0 --name infrared_classifier_eval_baseline
```

Evaluation outputs are written to `runs/eval/<name>/summary.json` and `runs/eval/<name>/summary.md`.

## Current baseline status

Latest completed runs:

- original visible detector baseline: `runs/train/visible_detector_baseline_cpu`
- boosted visible detector baseline: `runs/train/visible_detector_boosted_baseline_cpu`
- infrared classifier baseline: `runs/cls/infrared_classifier_baseline`
- original visible evaluations: `runs/eval/visible_detector_eval_baseline`, `runs/eval/visible_detector_eval_test_baseline`
- boosted visible evaluations: `runs/eval/visible_detector_boosted_eval_val`, `runs/eval/visible_detector_boosted_eval_test`
- infrared classifier evaluation: `runs/eval/infrared_classifier_eval_baseline`
- visible dataset analyses: `runs/analysis/dataset_1_defect_analysis`, `runs/analysis/dataset_1_defect_boosted_analysis`

Observed quality on current data:

- original visible detector test metrics: `precision=0.0`, `recall=0.0`, `mAP50=0.0`, `mAP50-95=0.0`
- boosted visible detector validation metrics: `precision=1.0`, `recall=0.5`, `mAP50=0.75`, `mAP50-95=0.675`
- boosted visible detector test metrics: `precision=1.0`, `recall=0.25`, `mAP50=0.625`, `mAP50-95=0.25`
- infrared classifier metrics: `top1=0.8`, `top5=1.0`

Interpretation:

- the visible detector is still not production-ready, but positive-sample boosting clearly moved it from a zero baseline to a usable proof-of-concept detector
- tuned visible postprocess improves demo cleanliness by removing many large, low-quality candidate boxes
- the infrared classifier remains the healthiest part of the project and is already suitable for demo / baseline reporting

## Useful scripts

- `scripts/train_yolo.py`: raw detection training
- `scripts/detect_and_refine.py`: raw detection inference with optional SAM refinement
- `scripts/train_yolo_cls.py`: raw classification training
- `scripts/predict_yolo_cls.py`: raw classification inference
- `scripts/run_pipeline.py`: registry-driven train / predict entry point
- `scripts/run_by_modality.py`: route by `visible/rgb` or `infrared/ir`
- `scripts/detect_postprocess.py`: shared visible detection postprocess and rendering helpers
- `scripts/web_demo.py`: local upload-and-visualize demo
- `scripts/launch_demo.cmd`: one-click Windows launcher for the demo
- `scripts/evaluate_pipelines.py`: unified evaluation for the current named pipelines
- `scripts/analyze_detection_dataset.py`: imbalance and box-size analysis for YOLO detection datasets
- `scripts/boost_detection_dataset.py`: create a boosted visible detection dataset by repeating positive training images

