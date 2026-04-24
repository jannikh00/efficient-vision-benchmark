import torch
from ultralytics import YOLO

from utils_voc import build_voc_subset, VOC_CLASSES
from metrics import SimpleTimer, compute_map


# ---------------- Configuration ----------------

VOC_ROOT = "./data/VOCdevkit"
YEAR = "2007"
SPLIT = "test"
SUBSET_SIZE = 200
SEED = 42
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


# ---------------- Device ----------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ---------------- Load same Pascal VOC subset ----------------

samples = build_voc_subset(
    voc_root=VOC_ROOT,
    year=YEAR,
    split=SPLIT,
    subset_size=SUBSET_SIZE,
    seed=SEED
)

print(f"Loaded Pascal VOC subset: {len(samples)} images")


# ---------------- Load YOLOv8 Nano ----------------

model = YOLO("yolov8n.pt")
print("Loaded pretrained YOLOv8-nano model.")


# ---------------- COCO to Pascal VOC label mapping ----------------

VOC_NAME_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}

COCO_TO_VOC_NAME = {
    "airplane": "aeroplane",
    "motorcycle": "motorbike",
    "couch": "sofa",
    "dining table": "diningtable",
    "tv": "tvmonitor",
    "potted plant": "pottedplant",
}


def map_yolo_label_to_voc_id(yolo_class_id):
    """
    Convert YOLO/COCO class id into Pascal VOC class id.
    Returns None if class is not part of VOC.
    """
    coco_name = model.names[int(yolo_class_id)]
    voc_name = COCO_TO_VOC_NAME.get(coco_name, coco_name)

    if voc_name not in VOC_NAME_TO_ID:
        return None

    return VOC_NAME_TO_ID[voc_name]


# ---------------- Inference + evaluation ----------------

predictions = []
ground_truths = []

timer = SimpleTimer()

for i, sample in enumerate(samples):
    if i % 25 == 0:
        print(f"Processing image {i}/{len(samples)}")

    image_path = sample["image_path"]

    # time only model inference
    timer.start()

    results = model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        device=device,
        verbose=False
    )

    timer.stop()

    result = results[0]

    pred_boxes = []
    pred_labels = []
    pred_scores = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu()
        labels = result.boxes.cls.cpu()
        scores = result.boxes.conf.cpu()

        for box, label, score in zip(boxes, labels, scores):
            voc_label = map_yolo_label_to_voc_id(int(label.item()))

            # skip COCO classes that are not in Pascal VOC
            if voc_label is None:
                continue

            pred_boxes.append(box.tolist())
            pred_labels.append(voc_label)
            pred_scores.append(float(score.item()))

    predictions.append({
        "image_id": sample["image_id"],
        "boxes": pred_boxes,
        "labels": pred_labels,
        "scores": pred_scores
    })

    ground_truths.append({
        "image_id": sample["image_id"],
        "boxes": sample["boxes"],
        "labels": sample["labels"]
    })


# ---------------- mAP calculation ----------------

class_ids = list(range(len(VOC_CLASSES)))

map_score, ap_per_class = compute_map(
    predictions=predictions,
    ground_truths=ground_truths,
    class_ids=class_ids,
    iou_threshold=IOU_THRESHOLD
)


# ---------------- Final outputs ----------------

print("\n===== YOLOv8-nano Evaluation Results =====")
print(f"Dataset: Pascal VOC {YEAR} {SPLIT}")
print(f"Subset size: {len(samples)} images")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print(f"IoU threshold for mAP: {IOU_THRESHOLD}")
print(f"Total inference time: {timer.total_seconds:.4f} seconds")
print(f"Average inference time: {timer.avg_ms():.4f} ms/image")
print(f"mAP@{IOU_THRESHOLD}: {map_score:.4f}")

print("\nAP per class:")
for class_id, ap in ap_per_class.items():
    class_name = VOC_CLASSES[class_id]
    print(f"{class_name}: {ap:.4f}")