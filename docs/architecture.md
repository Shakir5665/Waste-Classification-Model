# Architecture

## Overview

This project follows a **modular Python package** architecture that cleanly separates concerns between configuration, data, model, training, evaluation, and inference.

## Folder Structure

```
Waste_Sorting_System/
│
├── src/
│   └── waste_classifier/          # Core Python package
│       ├── __init__.py            # Package metadata
│       ├── config.py              # Loads config.yaml → CFG singleton
│       ├── data.py                # Data generators and loaders
│       ├── model.py               # CNN architecture definition
│       ├── train.py               # Training pipeline + model saving
│       ├── evaluate.py            # Evaluation framework
│       └── predict.py             # Inference pipeline
│
├── scripts/                       # CLI entry points
│   ├── train.py                   # Full training run
│   ├── evaluate.py                # Standalone evaluation of saved model
│   └── predict.py                 # Inference on a directory of images
│
├── dataset/                       # Dataset (train / val / test — all lowercase)
├── Realworld_data/                # Real-world inference images
│
├── outputs/
│   ├── models/                    # Saved model weights
│   └── reports/                   # Plots and classification report
│
├── notebooks/
│   └── colab_runner.ipynb         # Minimal Colab launcher (no logic)
│
├── docs/                          # Documentation
├── config.yaml                    # All configuration (single source of truth)
└── requirements.txt
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Loads `config.yaml` and exposes a `CFG` singleton used by all modules |
| `data.py` | Creates `ImageDataGenerator` instances and `flow_from_directory` flows |
| `model.py` | Defines and compiles the CNN architecture (`build_model()`) |
| `train.py` | Runs `model.fit()` with callbacks; saves the trained model |
| `evaluate.py` | Computes metrics, plots curves and confusion matrix, saves reports |
| `predict.py` | Loads images from a directory, runs inference, visualises and summarises |

## Data Flow

```
config.yaml
    ↓
CFG singleton (config.py)
    ↓
Data generators (data.py) ──────────────────────────┐
                                                     ↓
CNN architecture (model.py)  →  train.py (model.fit())
                                       ↓
                             outputs/models/waste_classifier.keras
                                       ↓
                             evaluate.py (metrics + plots)
                                       ↓
                             outputs/reports/ (PNG + TXT)
                                       ↓
                             predict.py (inference on new images)
```

## Design Principles

- **Single source of truth:** Every configurable value lives in `config.yaml`.  No hardcoded constants exist in source modules.
- **Separation of concerns:** Each module has one clear job.
- **Reproducibility:** Config is versioned alongside code.  Running `scripts/train.py` on the same data always produces the same result.
- **Non-interactive:** All scripts run headlessly — no Colab upload widgets in source code.
