import sys
sys.path.insert(0, 'scripts')
from run_pipeline import import_yolo, load_pipeline_config, resolve_path
import glob

YOLO = import_yolo()
pipeline = load_pipeline_config('visible_detector_boosted')
defaults = pipeline['default_predict']
weights = resolve_path(defaults['weights'])
print('weights:', weights)

model = YOLO(str(weights))
imgs = glob.glob('datasets/dataset_1_defect/images/test/*.jpg')
for img in imgs[:6]:
    r = model.predict(source=img, imgsz=416, conf=0.001, device='cpu', verbose=False)[0]
    h, w = r.orig_shape
    boxes = []
    for b in r.boxes:
        conf = round(float(b.conf), 4)
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        area = round((x2-x1)*(y2-y1)/(h*w), 5)
        boxes.append((conf, area))
    top = sorted(boxes, key=lambda x: -x[0])[:3]
    name = img.replace('\\', '/').split('/')[-1]
    print(name, '| top3 (conf, area_ratio):', top)
