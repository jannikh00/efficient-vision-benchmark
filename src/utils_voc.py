import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

from PIL import Image


# Pascal VOC classes in standard order
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# class name -> id
VOC_CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}


def get_voc_class_to_id() -> Dict[str, int]:
    """
    Return Pascal VOC class mapping.
    """
    return VOC_CLASS_TO_ID.copy()


def parse_voc_xml(xml_path: str) -> Dict:
    """
    Parse a Pascal VOC XML annotation file and return a raw dictionary.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []

    for obj in root.findall("object"):
        name = obj.findtext("name")
        difficult = obj.findtext("difficult")
        difficult = int(difficult) if difficult is not None else 0

        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.findtext("xmin")))
        ymin = int(float(bndbox.findtext("ymin")))
        xmax = int(float(bndbox.findtext("xmax")))
        ymax = int(float(bndbox.findtext("ymax")))

        objects.append({
            "name": name,
            "difficult": difficult,
            "bbox": [xmin, ymin, xmax, ymax]
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects
    }


def voc_annotation_to_common(xml_path: str) -> Dict:
    """
    Convert Pascal VOC XML annotation to a common detection format.

    Returns:
        {
            "boxes": [[xmin, ymin, xmax, ymax], ...],
            "labels": [class_id, ...],
            "label_names": [class_name, ...],
            "difficult": [0/1, ...],
            "width": int,
            "height": int,
            "filename": str
        }
    """
    parsed = parse_voc_xml(xml_path)

    boxes = []
    labels = []
    label_names = []
    difficult = []

    for obj in parsed["objects"]:
        class_name = obj["name"]

        # skip unknown labels just in case
        if class_name not in VOC_CLASS_TO_ID:
            continue

        boxes.append(obj["bbox"])
        labels.append(VOC_CLASS_TO_ID[class_name])
        label_names.append(class_name)
        difficult.append(obj["difficult"])

    return {
        "boxes": boxes,
        "labels": labels,
        "label_names": label_names,
        "difficult": difficult,
        "width": parsed["width"],
        "height": parsed["height"],
        "filename": parsed["filename"]
    }


def load_voc_sample(images_dir: str, annotations_dir: str, image_id: str) -> Dict:
    """
    Load one Pascal VOC sample using its image id (without extension).

    Example image_id: '000001'
    """
    image_path = os.path.join(images_dir, f"{image_id}.jpg")
    xml_path = os.path.join(annotations_dir, f"{image_id}.xml")

    image = Image.open(image_path).convert("RGB")
    target = voc_annotation_to_common(xml_path)

    return {
        "image_id": image_id,
        "image_path": image_path,
        "xml_path": xml_path,
        "image": image,
        "target": target
    }


def read_voc_image_ids(image_set_file: str) -> List[str]:
    """
    Read Pascal VOC split file, e.g. ImageSets/Main/train.txt or val.txt.
    """
    with open(image_set_file, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]
    return image_ids


def select_fixed_subset(items: List[str], subset_size: int, seed: int = 42) -> List[str]:
    """
    Select a fixed random subset from a list using a deterministic seed.
    """
    if subset_size >= len(items):
        return items.copy()

    rng = random.Random(seed)
    subset = items.copy()
    rng.shuffle(subset)
    return subset[:subset_size]


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert VOC box format to YOLO normalized format:
    (x_center, y_center, width, height)
    """
    box_width = xmax - xmin
    box_height = ymax - ymin
    x_center = xmin + box_width / 2.0
    y_center = ymin + box_height / 2.0

    x_center /= image_width
    y_center /= image_height
    box_width /= image_width
    box_height /= image_height

    return x_center, y_center, box_width, box_height


def write_yolo_label_file(xml_path: str, output_txt_path: str) -> None:
    """
    Convert one VOC XML annotation file into one YOLO label txt file.
    """
    target = voc_annotation_to_common(xml_path)

    lines = []

    for box, class_id in zip(target["boxes"], target["labels"]):
        xmin, ymin, xmax, ymax = box
        x_c, y_c, w, h = voc_box_to_yolo(
            xmin, ymin, xmax, ymax,
            target["width"], target["height"]
        )
        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def prepare_yolo_subset(
    image_ids: List[str],
    images_dir: str,
    annotations_dir: str,
    output_images_dir: str,
    output_labels_dir: str
) -> None:
    """
    Prepare a Pascal VOC subset in YOLO format.

    Copies image paths logically by filename expectation and writes YOLO label txt files.
    This function writes label files only. Image copying can be added if needed.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    for image_id in image_ids:
        xml_path = os.path.join(annotations_dir, f"{image_id}.xml")
        label_output_path = os.path.join(output_labels_dir, f"{image_id}.txt")
        write_yolo_label_file(xml_path, label_output_path)


def build_voc_subset(
    voc_root: str,
    year: str = "2007",
    split: str = "test",
    subset_size: int = 200,
    seed: int = 42
) -> List[Dict]:
    """
    Build a fixed Pascal VOC subset for evaluation.

    Returns:
        List of samples:
        {
            "image_id": str,
            "image_path": str,
            "image": PIL.Image,
            "boxes": [...],
            "labels": [...],
            "label_names": [...]
        }
    """

    # build paths
    voc_dir = os.path.join(voc_root, f"VOC{year}")
    images_dir = os.path.join(voc_dir, "JPEGImages")
    annotations_dir = os.path.join(voc_dir, "Annotations")
    image_set_file = os.path.join(voc_dir, "ImageSets", "Main", f"{split}.txt")

    # get all image ids
    image_ids = read_voc_image_ids(image_set_file)

    # select fixed subset
    subset_ids = select_fixed_subset(image_ids, subset_size=subset_size, seed=seed)

    samples = []

    for image_id in subset_ids:
        sample = load_voc_sample(images_dir, annotations_dir, image_id)

        # flatten target for easier use later
        samples.append({
            "image_id": sample["image_id"],
            "image_path": sample["image_path"],
            "image": sample["image"],
            "boxes": sample["target"]["boxes"],
            "labels": sample["target"]["labels"],
            "label_names": sample["target"]["label_names"]
        })

    return samples