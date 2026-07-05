# AI-Based Disaster Survivor Detection System

Real-time survivor detection in UAV/aerial disaster imagery using **YOLOv8**, built to support search-and-rescue (SAR) operations by automatically identifying survivors in drone footage.

## Overview

This project fine-tunes a YOLOv8 object detection model (via transfer learning from COCO) to detect human survivors in aerial disaster imagery. The pipeline covers dataset validation, model training, evaluation, CPU inference benchmarking, and result visualization — built to be usable in real-world, resource-constrained SAR deployment scenarios.

## Key Results

| Metric | Score |
|---|---|
| mAP@0.5 | 84.4% |
| mAP@0.5:0.95 | 58.3% |
| Precision | 87.9% |
| Recall | 79.9% |
| F1-Score | 83.7% |
| CPU Inference Latency | 120.9 ms/image |
| Throughput | 8.27 FPS (CPU) |

*Trained for 150 epochs at 640×640 resolution on a single-class ("survivor") dataset.*

## Architecture & Approach

- **Model:** YOLOv8 Medium, initialized with COCO-pretrained weights (transfer learning)
- **Framework:** Ultralytics YOLOv8 + PyTorch
- **Training:** 150 epochs, 640×640 image size, GPU-accelerated (A100)
- **Data Augmentation:** Albumentations pipeline — horizontal/vertical flips, rotation (±15°), perspective distortion, brightness/contrast jitter (to improve robustness to varied lighting/altitude conditions typical of UAV footage)
- **Evaluation:** Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95 on a held-out test split, plus a confusion matrix and GT-vs-prediction visualizations
- **Inference Benchmarking:** Measured real-world CPU latency/FPS to validate feasibility for edge/low-resource deployment (e.g., onboard drone hardware)

## Project Pipeline

1. Dataset integrity check & auto-detection of class labels
2. YOLO-format `data.yaml` generation
3. Augmentation preview and pipeline setup
4. Model training (YOLOv8m, transfer learning)
5. Evaluation on test split (Precision/Recall/mAP/F1)
6. CPU inference latency & FPS benchmarking
7. Confusion matrix generation
8. Ground truth vs. prediction visualization
9. Training curve plots (loss & mAP over epochs)
10. Final results summary report

## Tech Stack

`Python` · `Ultralytics YOLOv8` · `PyTorch` · `OpenCV` · `Albumentations` · `Matplotlib` · `Google Colab (A100 GPU)`

## Repository Structure

```
├── SAR_Detection_YOLO_Notebook.ipynb   # Main notebook (end-to-end pipeline)
├── data.yaml                           # YOLO dataset configuration
├── runs/                               # Training run outputs & weights
├── outputs/                            # Metrics, plots, benchmark results
└── README.md
```

## How to Run

1. Open the notebook in Google Colab (GPU runtime recommended)
2. Mount your dataset (update dataset path in the configuration cell)
3. Run cells sequentially — the notebook handles dependency installation, data validation, training, and evaluation
4. Trained weights and evaluation outputs are saved automatically to the output directory

## Future Improvements

- Expand to multi-class detection (e.g., differentiate survivors by mobility status/urgency)
- Model quantization/pruning for faster edge deployment
- Real-time video stream integration for live drone feeds

## Team

- Rida Imtiaz
- Fatima Shaheen

## License

This project was developed for academic purposes.
