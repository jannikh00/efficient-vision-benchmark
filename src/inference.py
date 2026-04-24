import os
import cv2
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.transforms import functional as F

from utils_voc import build_voc_subset, VOC_CLASSES


# ---------------- Configuration ----------------

VOC_ROOT = "./data/VOCdevkit"
YEAR = "2007"
SPLIT = "test"
SUBSET_SIZE = 2
SEED = 7

CONF_THRESHOLD = 0.5
OUTPUT_DIR = "./results/inference"


# ---------------- Device ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- Load two Pascal VOC test images ----------------

samples = build_voc_subset(
    voc_root=VOC_ROOT,
    year=YEAR,
    split=SPLIT,
    subset_size=SUBSET_SIZE,
    seed=SEED,
)

print(f"Loaded {len(samples)} images for inference.")


# ---------------- Load Faster R-CNN ----------------

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

model.to(device)
model.eval()

print("Loaded pretrained Faster R-CNN model.")


# ---------------- COCO to Pascal VOC label mapping ----------------

coco_categories = weights.meta["categories"]
VOC_NAME_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}

COCO_TO_VOC_NAME = {
    "airplane": "aeroplane",
    "motorcycle": "motorbike",
    "couch": "sofa",
    "dining table": "diningtable",
    "tv": "tvmonitor",
    "potted plant": "pottedplant",
}


def map_coco_label_to_voc_name(coco_label_id):
    """
    Convert COCO class id from Faster R-CNN into Pascal VOC class name.
    Returns None if the class is not part of Pascal VOC.
    """
    coco_name = coco_categories[coco_label_id]
    voc_name = COCO_TO_VOC_NAME.get(coco_name, coco_name)

    if voc_name not in VOC_NAME_TO_ID:
        return None

    return voc_name


# ---------------- Drawing helper ----------------

def draw_detection(image_bgr, box, label_name, score):
    """
    Draw one bounding box, class label, and confidence score.
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    # draw box
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # label text
    text = f"{label_name}: {score:.2f}"

    # text background
    text_size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )

    text_width, text_height = text_size

    cv2.rectangle(
        image_bgr,
        (x1, max(0, y1 - text_height - 8)),
        (x1 + text_width + 4, y1),
        (0, 255, 0),
        -1
    )

    # text
    cv2.putText(
        image_bgr,
        text,
        (x1 + 2, max(12, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )


# ---------------- Run inference and save outputs ----------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

with torch.no_grad():
    for idx, sample in enumerate(samples):
        image_pil = sample["image"]
        image_path = sample["image_path"]

        # convert PIL image to tensor
        image_tensor = F.to_tensor(image_pil).to(device)

        # run Faster R-CNN
        output = model([image_tensor])[0]

        # read original image with OpenCV for drawing
        image_bgr = cv2.imread(image_path)

        detections = []

        boxes = output["boxes"].detach().cpu()
        labels = output["labels"].detach().cpu()
        scores = output["scores"].detach().cpu()

        for box, label, score in zip(boxes, labels, scores):
            score_value = float(score.item())

            # confidence filter
            if score_value < CONF_THRESHOLD:
                continue

            label_name = map_coco_label_to_voc_name(int(label.item()))

            # skip non-VOC classes
            if label_name is None:
                continue

            box_list = box.tolist()

            draw_detection(
                image_bgr=image_bgr,
                box=box_list,
                label_name=label_name,
                score=score_value
            )

            detections.append({
                "label": label_name,
                "score": score_value,
                "box": box_list
            })

        # save result image
        output_path = os.path.join(
            OUTPUT_DIR,
            f"image{idx + 1}_detected.jpg"
        )

        cv2.imwrite(output_path, image_bgr)

        # console summary
        print(f"\nImage {idx + 1}: {sample['image_id']}")
        print(f"Saved to: {output_path}")
        print(f"Number of detections above threshold: {len(detections)}")

        for det in detections[:5]:
            print(
                f"  {det['label']} "
                f"score={det['score']:.2f} "
                f"box={[round(v, 1) for v in det['box']]}"
            )

print("\nInference visualization completed.")