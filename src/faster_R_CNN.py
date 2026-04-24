import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

from utils_voc import build_voc_subset, VOC_CLASSES
from metrics import SimpleTimer, compute_map


# ---------------- Configuration ----------------

VOC_ROOT = "./data/VOCdevkit"   # expected: ./data/VOCdevkit/VOC2007/...
YEAR = "2007"
SPLIT = "test"
SUBSET_SIZE = 200
SEED = 42
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


# ---------------- Device ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- Load Pascal VOC subset ----------------

samples = build_voc_subset(
    voc_root=VOC_ROOT,
    year=YEAR,
    split=SPLIT,
    subset_size=SUBSET_SIZE,
    seed=SEED
)

print(f"Loaded Pascal VOC subset: {len(samples)} images")


# ---------------- Load Faster R-CNN ----------------

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

model.to(device)
model.eval()

print("Loaded pretrained Faster R-CNN model.")


# ---------------- Class mapping ----------------
# Torchvision COCO labels use class names, Pascal VOC uses class ids.
# We map predicted COCO class names to Pascal VOC class ids when possible.

coco_categories = weights.meta["categories"]

VOC_NAME_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}

# Some COCO/VOC names differ slightly
COCO_TO_VOC_NAME = {
    "airplane": "aeroplane",
    "motorcycle": "motorbike",
    "couch": "sofa",
    "dining table": "diningtable",
    "tv": "tvmonitor",
    "potted plant": "pottedplant",
}


def map_coco_label_to_voc_id(coco_label_id):
    """
    Convert a COCO label id from Faster R-CNN into a Pascal VOC class id.
    Returns None if the class does not exist in VOC.
    """
    coco_name = coco_categories[coco_label_id]

    # convert COCO name to matching VOC name if needed
    voc_name = COCO_TO_VOC_NAME.get(coco_name, coco_name)

    if voc_name not in VOC_NAME_TO_ID:
        return None

    return VOC_NAME_TO_ID[voc_name]


# ---------------- Inference + evaluation ----------------

predictions = []
ground_truths = []

timer = SimpleTimer()

with torch.no_grad():
    for i, sample in enumerate(samples):
        if i % 25 == 0:
            print(f"Processing image {i}/{len(samples)}")

        image = sample["image"]

        # Convert PIL image to tensor [C, H, W]
        image_tensor = F.to_tensor(image).to(device)

        # Faster R-CNN expects a list of image tensors
        timer.start()
        output = model([image_tensor])[0]
        timer.stop()

        pred_boxes = []
        pred_labels = []
        pred_scores = []

        boxes = output["boxes"].detach().cpu()
        labels = output["labels"].detach().cpu()
        scores = output["scores"].detach().cpu()

        for box, label, score in zip(boxes, labels, scores):
            score_value = float(score.item())

            # confidence threshold
            if score_value < CONF_THRESHOLD:
                continue

            # map COCO label to VOC label
            voc_label = map_coco_label_to_voc_id(int(label.item()))

            # skip non-VOC classes
            if voc_label is None:
                continue

            pred_boxes.append(box.tolist())
            pred_labels.append(voc_label)
            pred_scores.append(score_value)

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

print("\n===== Faster R-CNN Evaluation Results =====")
print(f"Dataset: Pascal VOC {YEAR} {SPLIT}")
print(f"Subset size: {len(samples)} images")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print(f"IoU threshold for mAP: {IOU_THRESHOLD}")
print(f"Total inference time: {timer.total_seconds:.4f} seconds")
print(f"Average inference time: {timer.avg_ms():.4f} ms/image")
print(f"mAP@{IOU_THRESHOLD}: {map_score:.4f}")

print("\nAP per class:")
for_class_lines = []
for class_id, ap in ap_per_class.items():
    class_name = VOC_CLASSES[class_id]
    for_class_lines.append(f"{class_name}: {ap:.4f}")

for line in for_class_lines:
    print(line)