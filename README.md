# Efficient Vision Benchmark
Benchmarking CNNs and Spiking Neural Networks (SNNs) for Efficient Computer Vision

## 🚀 Overview
This project explores the trade-offs between traditional Convolutional Neural Networks (CNNs) and Spiking Neural Networks (SNNs) in computer vision tasks.

While CNNs achieve high accuracy, they are computationally expensive. SNNs offer a biologically inspired, event-driven alternative that can significantly reduce energy consumption.

This repository benchmarks both approaches across:
- Image classification (CIFAR-10)
- ANN → SNN conversion
- Direct SNN training with surrogate gradients
- Object detection (Faster R-CNN vs YOLO)

---

## 🎯 Objectives
- Compare accuracy, latency, and efficiency between CNNs and SNNs
- Analyze the impact of ANN → SNN conversion
- Evaluate directly trained SNNs using surrogate gradients
- Benchmark modern object detection models

---

## 🧱 Project Structure

#### src/
cnn.py                # CNN baseline model (CIFAR-10)
ann_snn.py            # ANN → SNN conversion (rate coding)
surrogate_snn.py      # Direct SNN training (LIF + BPTT)
faster_rcnn.py        # Faster R-CNN inference
yolo.py               # YOLOv8 inference + fine-tuning
inference.py          # Visualization of detection results

#### utils/
dataset.py
metrics.py
visualization.py

#### results/
plots/
logs/
models/

---

## Methods

### 1. CNN Baseline
- Custom CNN trained from scratch on CIFAR-10
- Achieves >80% test accuracy
- Serves as baseline for comparison

### 2. ANN → SNN Conversion
- Converts trained CNN into SNN using rate coding
- Evaluates temporal dynamics (time steps T)
- Measures spike activity and efficiency

### 3. Direct SNN Training
- Leaky Integrate-and-Fire (LIF) neurons
- Backpropagation Through Time (BPTT)
- Surrogate gradient optimization

### 4. Object Detection
- Faster R-CNN (ResNet50-FPN)
- YOLOv8 (Ultralytics)
- Metrics: mAP and inference time
- YOLO fine-tuning on Pascal VOC subset

---

## Key Metrics

| Metric        | CNN        | Converted SNN | Trained SNN |
|--------------|-----------|--------------|-------------|
| Accuracy     |       |   |   |
| Latency      |        |    |   |
| Efficiency   | | | |
| Sparsity     |       |         |        |

---

## Key Insights



---

## Tech Stack

- PyTorch
- snnTorch
- Ultralytics (YOLOv8)
- OpenCV
- NumPy / Matplotlib

---

## Results

- --- put plots here ---
- Training/validation curves
- Accuracy vs time steps
- Detection visualizations

---

## Takeaway
