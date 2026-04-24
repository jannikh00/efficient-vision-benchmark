import time
from typing import Dict, List, Tuple


class SimpleTimer:
    """
    Simple timing helper for inference.
    """
    def __init__(self):
        self.start_time = None
        self.total_seconds = 0.0
        self.count = 0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        elapsed = time.perf_counter() - self.start_time
        self.total_seconds += elapsed
        self.count += 1
        self.start_time = None
        return elapsed

    def avg_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return (self.total_seconds / self.count) * 1000.0


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute IoU between two boxes in [xmin, ymin, xmax, ymax] format.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    """
    Compute AP using all-point interpolation.
    """
    recalls = [0.0] + recalls + [1.0]
    precisions = [0.0] + precisions + [0.0]

    # make precision monotonically decreasing from right to left
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        delta_recall = recalls[i] - recalls[i - 1]
        ap += delta_recall * precisions[i]

    return ap


def compute_class_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_id: int,
    iou_threshold: float = 0.5
) -> float:
    """
    Compute AP for one class.

    predictions format:
        [
            {
                "image_id": str,
                "boxes": [[...], ...],
                "labels": [int, ...],
                "scores": [float, ...]
            },
            ...
        ]

    ground_truths format:
        [
            {
                "image_id": str,
                "boxes": [[...], ...],
                "labels": [int, ...]
            },
            ...
        ]
    """
    gt_by_image = {}
    total_gt = 0

    for item in ground_truths:
        image_id = item["image_id"]
        gt_boxes = []

        for box, label in zip(item["boxes"], item["labels"]):
            if label == class_id:
                gt_boxes.append({
                    "box": box,
                    "matched": False
                })

        gt_by_image[image_id] = gt_boxes
        total_gt += len(gt_boxes)

    if total_gt == 0:
        return 0.0

    class_predictions = []

    for item in predictions:
        image_id = item["image_id"]

        for box, label, score in zip(item["boxes"], item["labels"], item["scores"]):
            if label == class_id:
                class_predictions.append({
                    "image_id": image_id,
                    "box": box,
                    "score": score
                })

    class_predictions.sort(key=lambda x: x["score"], reverse=True)

    tp = []
    fp = []

    for pred in class_predictions:
        image_id = pred["image_id"]
        pred_box = pred["box"]

        gt_candidates = gt_by_image.get(image_id, [])

        best_iou = 0.0
        best_idx = -1

        for idx, gt in enumerate(gt_candidates):
            if gt["matched"]:
                continue

            iou = compute_iou(pred_box, gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx != -1:
            gt_candidates[best_idx]["matched"] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    cum_tp = []
    cum_fp = []

    running_tp = 0
    running_fp = 0

    for t, f in zip(tp, fp):
        running_tp += t
        running_fp += f
        cum_tp.append(running_tp)
        cum_fp.append(running_fp)

    recalls = []
    precisions = []

    for t, f in zip(cum_tp, cum_fp):
        recall = t / total_gt
        precision = t / (t + f) if (t + f) > 0 else 0.0
        recalls.append(recall)
        precisions.append(precision)

    return compute_ap(recalls, precisions)


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_ids: List[int],
    iou_threshold: float = 0.5
) -> Tuple[float, Dict[int, float]]:
    """
    Compute mean AP over the provided class ids.
    """
    ap_per_class = {}

    for class_id in class_ids:
        ap = compute_class_ap(
            predictions=predictions,
            ground_truths=ground_truths,
            class_id=class_id,
            iou_threshold=iou_threshold
        )
        ap_per_class[class_id] = ap

    if len(ap_per_class) == 0:
        return 0.0, ap_per_class

    mean_ap = sum(ap_per_class.values()) / len(ap_per_class)
    return mean_ap, ap_per_class