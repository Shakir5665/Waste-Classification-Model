# ♻️ Waste Sorting System

> Binary image classification of household waste into **Organic** and **Recyclable** categories using a custom Convolutional Neural Network.

**Team TECH DREAMERS** — ICT 3212 Intelligent Systems
**Test Accuracy: 95.30%** on 1,680 completely unseen images

> 📖 **New here?** Start with the [Complete Step-by-Step Guide](docs/GUIDE.md) — it walks you through connecting Colab, training the model, and testing with real-world images.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Quick Start](#quick-start)
4. [Google Colab Workflow](#google-colab-workflow)
5. [Dataset](#dataset)
6. [Model Architecture](#model-architecture)
7. [Training & Regularization Strategy](#training--regularization-strategy)
8. [Results](#results)
9. [Configuration](#configuration)
10. [Documentation](#documentation)

---

## Project Overview

This project implements a deep learning pipeline to automate the visual classification of household waste for smart bin sorting systems. The model distinguishes between **Organic** waste (food scraps, plant matter) and **Recyclable** waste (plastic, metal, glass, paper).

The repository is structured as a professional, modular Python codebase — **not** a notebook-centric project. All logic lives in importable Python modules. The included Colab notebook is a minimal launcher that simply runs the training script on a free GPU runtime.

---

## Repository Structure

```
Waste_Sorting_System/
│
├── src/
│   └── waste_classifier/          # Core application package
│       ├── __init__.py
│       ├── config.py              # Loads config.yaml → CFG singleton
│       ├── data.py                # Data generators and loaders
│       ├── model.py               # CNN architecture
│       ├── train.py               # Training pipeline
│       ├── evaluate.py            # Evaluation framework
│       └── predict.py             # Inference pipeline
│
├── scripts/                       # CLI entry points
│   ├── train.py                   # Full training run
│   ├── evaluate.py                # Evaluate a saved model
│   └── predict.py                 # Inference on a directory
│
├── dataset/                       # Structured dataset (Train/Validation/Test)
├── Realworld_data/                # Real-world inference images
│
├── outputs/
│   ├── models/                    # Saved model weights (.keras)
│   └── reports/                   # Training curves, confusion matrix, report
│
├── notebooks/
│   └── colab_runner.ipynb         # Minimal Colab GPU launcher
│
├── docs/
│   ├── architecture.md
│   ├── training.md
│   └── inference.md
│
├── config.yaml                    # All configuration (single source of truth)
└── requirements.txt
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Shakir5665/Mini-project---Waste-Sorting-System.git
cd Mini-project---Waste-Sorting-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python scripts/train.py
```

This runs the complete pipeline: data loading → model build → training → evaluation → saved outputs.

### 4. Run inference on real-world images

```bash
python scripts/predict.py --input Realworld_data/
```

### 5. Evaluate a saved model independently

```bash
python scripts/evaluate.py
```

---

## Google Colab Workflow

Open [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) in Google Colab with a **T4 GPU** runtime.

The notebook has **6 cells only** — each with a single responsibility:

| Step | Cell | Action |
|---|---|---|
| 1 | Clone | Clones / pulls the GitHub repository |
| 2 | Install | `pip install -r requirements.txt` |
| 3 | Upload | Upload the dataset ZIP (extracted automatically) |
| 4 | **Train** | `!python scripts/train.py` |
| 5 | Predict | `!python scripts/predict.py` (optional) |
| 6 | Download | Downloads model and reports (optional) |

The notebook contains **no application logic**. It is simply a free GPU runtime.

---

## Dataset

```
dataset/
├── train/         21,857 images  (used to train model weights)
│   ├── organic/
│   └── recyclable/
├── val/            2,471 images  (used for early stopping / hyperparameter tuning)
│   ├── organic/
│   └── recyclable/
└── test/           2,470 images  (completely unseen — final evaluation only)
    ├── organic/
    └── recyclable/
```

- **Classes:** `organic` (label 0) · `recyclable` (label 1)
- **Image size:** 128 × 128 pixels, 3-channel RGB
- **Total:** 26,798 images

---

## Model Architecture

A custom CNN with three convolutional blocks followed by a fully-connected head with dropout regularisation:

```
Input (128, 128, 3)
    ↓
Conv2D(32, 3×3, relu) → MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3, relu) → MaxPooling2D(2×2)
    ↓
Conv2D(128, 3×3, relu) → MaxPooling2D(2×2)
    ↓
Flatten
    ↓
Dense(128, relu) → Dropout(0.5)
    ↓
Dense(1, sigmoid)
```

**Total parameters:** 3,304,769 (12.61 MB)  
**Optimizer:** Adam · **Loss:** Binary crossentropy

---

## Training & Regularization Strategy

| Technique | Setting | Purpose |
|---|---|---|
| Data augmentation | Rotation ±20°, zoom 0.2, shear 0.2, horizontal flip | Reduces overfitting; improves generalisation |
| Dropout | Rate = 0.5 (before output) | Prevents co-adaptation of neurons |
| Early Stopping | Monitor `val_loss`, patience=3, restore best weights | Prevents overtraining |
| Explicit test split | 1,680 unseen images | True generalisation measurement |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **95.30%** |
| Test Loss | 0.1412 |

### Classification Report

```
              precision    recall  f1-score   support

     Organic       0.68      0.81      0.74       138
  Recyclable       0.98      0.97      0.97      1542

    accuracy                           0.95      1680
   macro avg       0.83      0.89      0.86      1680
weighted avg       0.96      0.95      0.95      1680
```

All evaluation artefacts (training curves, confusion matrix heatmap, classification report) are automatically saved to `outputs/reports/` when running the training script.

---

## Configuration

All configurable parameters live in [`config.yaml`](config.yaml).  
**No hardcoded values exist in any source module.**

```yaml
data:
  img_size: [128, 128]
  batch_size: 32

training:
  epochs: 20
  optimizer: "adam"

callbacks:
  early_stopping:
    patience: 3
    restore_best_weights: true

model:
  filters: [32, 64, 128]
  dropout_rate: 0.5
```

---

## Documentation

| Document | Description |
|---|---|
| [`docs/GUIDE.md`](docs/GUIDE.md) | **Complete step-by-step guide** — Colab setup, training, real-world testing |
| [`docs/learning.md`](docs/learning.md) | **Learning reference** — every technology, library, and technique explained |
| [`docs/architecture.md`](docs/architecture.md) | Repository structure, module map, data flow |
| [`docs/training.md`](docs/training.md) | Training pipeline, configuration, expected outputs |
| [`docs/inference.md`](docs/inference.md) | Inference pipeline, threshold, adding new images |

---

## Team Members

Developed as part of the ICT 3212 — Intelligent Systems coursework.

**Team TECH DREAMERS**

---

_For questions about the architecture or implementation, refer to the [`docs/`](docs/) directory._
