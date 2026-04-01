"""Train YOLOv8s on the boosted visible dataset."""
import sys
sys.path.insert(0, 'scripts')
from common import prepare_ultralytics_env
prepare_ultralytics_env()

from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # start from nano, upgrade to s if download available

results = model.train(
    data='configs/dataset_1_defect_boosted.yaml',
    epochs=80,
    imgsz=512,
    batch=1,
    workers=0,
    device='cpu',
    patience=20,
    lr0=0.005,
    lrf=0.01,
    mosaic=1.0,
    flipud=0.3,
    fliplr=0.5,
    degrees=5.0,
    project='runs/train',
    name='visible_detector_v3',
    exist_ok=True,
)
print('Training complete:', results)
