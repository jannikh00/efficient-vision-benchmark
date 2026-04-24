import os
import shutil
import time

import torch
from ultralytics import YOLO

from utils_voc import (
    VOC_CLASSES,
    read_voc_image_ids,
    select_fixed_subset,
    write_yolo_label_file,
    build_voc_subset,
)
from metrics import SimpleTimer, compute_map


# ---------------- Configuration ----------------

VOC_ROOT = "./data/VOCdevkit"
YEAR = "2007"

TRAINVAL_SPLIT = "trainval"
TEST_SPLIT = "test"

TOTAL_TRAIN_IMAGES = 1200
TRAIN_IMAGES = 1000
VAL_IMAGES = 200

SEED = 42
EPOCHS = 3
IMG_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

YOLO_DATASET_DIR = "./data/yolo_voc_subset"
YOLO_YAML_PATH = os.path.join(YOLO_DATASET_DIR, "voc_subset.yaml")


# ---------------- Device ----------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ---------------- COCO to VOC mapping for pretrained YOLO ----------------

VOC_NAME_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}

COCO_TO_VOC_NAME = {
    "airplane": "aeroplane",
    "motorcycle": "motorbike",
    "couch": "sofa",
    "dining table": "diningtable",
    "tv": "tvmonitor",
    "potted plant": "pottedplant",
}


def map_coco_yolo_label_to_voc_id(model, yolo_class_id):
    """
    Convert pretrained YOLO/COCO class id into Pascal VOC class id.
    Used before fine-tuning.
    """
    coco_name = model.names[int(yolo_class_id)]
    voc_name = COCO_TO_VOC_NAME.get(coco_name, coco_name)

    if voc_name not in VOC_NAME_TO_ID:
        return None

    return VOC_NAME_TO_ID[voc_name]


def map_finetuned_label_to_voc_id(yolo_class_id):
    """
    Fine-tuned model is trained directly on VOC labels 0-19.
    """
    yolo_class_id = int(yolo_class_id)

    if 0 <= yolo_class_id < len(VOC_CLASSES):
        return yolo_class_id

    return None


# ---------------- Dataset preparation ----------------

def prepare_yolo_train_val_dataset():
    """
    Convert Pascal VOC trainval subset into YOLO format.

    Output structure:
        data/yolo_voc_subset/
            images/train/
            images/val/
            labels/train/
            labels/val/
            voc_subset.yaml
    """

    voc_dir = os.path.join(VOC_ROOT, f"VOC{YEAR}")
    images_dir = os.path.join(voc_dir, "JPEGImages")
    annotations_dir = os.path.join(voc_dir, "Annotations")
    split_file = os.path.join(voc_dir, "ImageSets", "Main", f"{TRAINVAL_SPLIT}.txt")

    all_ids = read_voc_image_ids(split_file)

    subset_ids = select_fixed_subset(
        all_ids,
        subset_size=TOTAL_TRAIN_IMAGES,
        seed=SEED
    )

    train_ids = subset_ids[:TRAIN_IMAGES]
    val_ids = subset_ids[TRAIN_IMAGES:TRAIN_IMAGES + VAL_IMAGES]

    folders = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]

    for folder in folders:
        os.makedirs(os.path.join(YOLO_DATASET_DIR, folder), exist_ok=True)

    def copy_images_and_labels(image_ids, split_name):
        for image_id in image_ids:
            src_image = os.path.join(images_dir, f"{image_id}.jpg")
            dst_image = os.path.join(YOLO_DATASET_DIR, "images", split_name, f"{image_id}.jpg")

            src_xml = os.path.join(annotations_dir, f"{image_id}.xml")
            dst_label = os.path.join(YOLO_DATASET_DIR, "labels", split_name, f"{image_id}.txt")

            shutil.copy2(src_image, dst_image)
            write_yolo_label_file(src_xml, dst_label)

    print("Preparing YOLO-format dataset...")
    copy_images_and_labels(train_ids, "train")
    copy_images_and_labels(val_ids, "val")

    yaml_content = f"""path: {os.path.abspath(YOLO_DATASET_DIR)}
train: images/train
val: images/val

nc: {len(VOC_CLASSES)}
names:
"""

    for class_name in VOC_CLASSES:
        yaml_content += f"  - {class_name}\n"

    with open(YOLO_YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"YOLO dataset prepared at: {YOLO_DATASET_DIR}")
    print(f"Dataset YAML written to: {YOLO_YAML_PATH}")


# ---------------- Custom mAP evaluation ----------------

def evaluate_yolo_on_voc_subset(model, samples, use_coco_mapping):
    """
    Evaluate YOLO model on VOC samples using the same custom mAP logic
    as the pretrained YOLO.py evaluation.

    use_coco_mapping=True:
        pretrained YOLOv8n.pt uses COCO class ids

    use_coco_mapping=False:
        fine-tuned model uses VOC class ids directly
    """

    predictions = []
    ground_truths = []

    timer = SimpleTimer()

    for i, sample in enumerate(samples):
        if i % 25 == 0:
            print(f"Evaluating image {i}/{len(samples)}")

        image_path = sample["image_path"]

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
                if use_coco_mapping:
                    voc_label = map_coco_yolo_label_to_voc_id(model, int(label.item()))
                else:
                    voc_label = map_finetuned_label_to_voc_id(int(label.item()))

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

    map_score, ap_per_class = compute_map(
        predictions=predictions,
        ground_truths=ground_truths,
        class_ids=list(range(len(VOC_CLASSES))),
        iou_threshold=IOU_THRESHOLD
    )

    return map_score, ap_per_class, timer


# ---------------- Main script ----------------

if __name__ == "__main__":

    # 1. Prepare YOLO dataset for fine-tuning
    prepare_yolo_train_val_dataset()

    # 2. Build fixed VOC test subset for before/after mAP comparison
    eval_samples = build_voc_subset(
        voc_root=VOC_ROOT,
        year=YEAR,
        split=TEST_SPLIT,
        subset_size=200,
        seed=SEED
    )

    print(f"\nLoaded evaluation subset: {len(eval_samples)} images")

    # 3. Load pretrained YOLOv8-nano
    pretrained_model = YOLO("yolov8n.pt")
    print("\nLoaded pretrained YOLOv8-nano model.")

    # 4. Evaluate before fine-tuning
    print("\n===== Baseline Evaluation Before Fine-Tuning =====")

    before_map, before_ap_per_class, before_timer = evaluate_yolo_on_voc_subset(
        model=pretrained_model,
        samples=eval_samples,
        use_coco_mapping=True
    )

    print(f"Before fine-tuning mAP@{IOU_THRESHOLD}: {before_map:.4f}")
    print(f"Before fine-tuning avg inference time: {before_timer.avg_ms():.4f} ms/image")

    # 5. Fine-tune YOLO
    print("\n===== Fine-Tuning YOLOv8-nano =====")

    train_start = time.perf_counter()

    train_results = pretrained_model.train(
        data=YOLO_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=8,
        device=device,
        workers=0,
        project="results/yolo_finetune",
        name="voc_yolov8n",
        exist_ok=True
    )

    train_end = time.perf_counter()
    training_time = train_end - train_start

    print(f"\nTraining time: {training_time:.2f} seconds")

    # 6. Load best fine-tuned model
    best_model_path = os.path.join(str(train_results.save_dir), "weights", "best.pt")
    finetuned_model = YOLO(best_model_path)

    print(f"Loaded fine-tuned YOLO model from: {best_model_path}")

    # 7. Evaluate after fine-tuning
    print("\n===== Evaluation After Fine-Tuning =====")

    after_map, after_ap_per_class, after_timer = evaluate_yolo_on_voc_subset(
        model=finetuned_model,
        samples=eval_samples,
        use_coco_mapping=False
    )

    print(f"After fine-tuning mAP@{IOU_THRESHOLD}: {after_map:.4f}")
    print(f"After fine-tuning avg inference time: {after_timer.avg_ms():.4f} ms/image")

    # 8. Final comparison summary
    improvement = after_map - before_map

    print("\n===== YOLO Fine-Tuning Summary =====")
    print(f"Training images: {TRAIN_IMAGES}")
    print(f"Validation images for training: {VAL_IMAGES}")
    print(f"Fine-tuning epochs: {EPOCHS}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Before mAP@{IOU_THRESHOLD}: {before_map:.4f}")
    print(f"After mAP@{IOU_THRESHOLD}: {after_map:.4f}")
    print(f"mAP improvement: {improvement:.4f}")

    print("\nClass-wise AP before vs after:")
    for class_id, class_name in enumerate(VOC_CLASSES):
        before_ap = before_ap_per_class.get(class_id, 0.0)
        after_ap = after_ap_per_class.get(class_id, 0.0)
        diff = after_ap - before_ap
        print(f"{class_name}: before={before_ap:.4f}, after={after_ap:.4f}, change={diff:.4f}")