# Efficient Vision Benchmark

CNN vs SNN + Faster R-CNN vs YOLO (focus: accuracy, latency, efficiency)

## Overview
This project compares:
- CNN vs SNN (classification on CIFAR-10)
- Faster R-CNN vs YOLOv8 (object detection on Pascal VOC)

Main goal: understand trade-offs (accuracy vs speed vs sparsity)

## Structure

src/
├── cnn.py  
├── ann_snn.py  
├── surrogate_snn.py  
├── faster_R_CNN.py  
├── YOLO.py  
├── YOLO_finetune.py  
├── inference.py  
├── utils_voc.py  
└── metrics.py  

## Results

### Classification

| Model | Accuracy |
|---|---:|
| CNN | 81.46% |
| SNN (converted / trained) | lower than CNN |

- CNN works best
- SNN performance depends heavily on training and conversion setup

### Object Detection

| Model | mAP@0.5 | Time |
|---|---:|---:|
| Faster R-CNN | 0.7381 | 1054 ms |
| YOLOv8-nano | 0.6071 | 41 ms |

- Faster R-CNN -> better accuracy  
- YOLO -> much faster  

## Run

pip install -r requirements.txt

python src/cnn.py  
python src/ann_snn.py  
python src/surrogate_snn.py  
python src/faster_R_CNN.py  
python src/YOLO.py  
python src/inference.py  
python src/YOLO_finetune.py  

## Notes

data/  
results/  
runs/  
*.pt  
.venv/  

not included (too large)

## Takeaway

- CNN -> best accuracy  
- SNN -> more efficient but harder to train well  
- Faster R-CNN -> accurate but slow  
- YOLO -> fast but less accurate  
