# Waste Classification Model

A production-ready image classification pipeline for identifying organic and recyclable waste. This project demonstrates a complete end-to-end machine learning workflow with clean architecture, configurable training, evaluation, and inference.

## 🚀 Project Overview

A waste classification system built with TensorFlow that can:

- Train a convolutional neural network on image data
- Evaluate model performance with test metrics and visualizations
- Run real-world image inference from a directory of photos
- Support resume training, checkpointing, and Google Drive sync

The project is designed for recruiters and technical reviewers who want to assess clear engineering practices, reproducibility, and model deployment readiness.

## 🔧 Key Features

- Modular Python package under `src/waste_classifier`
- Centralized hyperparameter configuration via `config.yaml`
- Data pipelines using directory-based image flows
- CNN model architecture with dropout and configurable filters
- Early stopping, checkpoint saving, and best-model selection
- Evaluation artifacts: training curves, confusion matrices, ROC/PR curves, classification report
- Real-world inference support via `scripts/predict.py`
- Google Colab integration through `notebooks/colab_runner.ipynb`

## 📁 Repository Structure

- `config.yaml` — all configurable options for data, augmentation, training, model, outputs, and inference
- `requirements.txt` — Python dependencies for training and evaluation
- `scripts/` — command-line entrypoints for training, evaluation, and prediction
- `src/waste_classifier/` — reusable source code modules
- `dataset/` — dataset split into `train/`, `val/`, and `test/`
- `Realworld_data/` — sample real-world images for inference
- `docs/` — usage guides, architecture notes, training instructions
- `outputs/` — generated models and reports after training

## 📦 Technology Stack

- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- PyYAML
- Pillow

## ▶️ Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure the dataset follows this structure:

```text
dataset/
  train/
    organic/
    recyclable/
  val/
    organic/
    recyclable/
  test/
    organic/
    recyclable/
```

3. Train the model:

```bash
python scripts/train.py
```

4. Evaluate the trained model:

```bash
python scripts/evaluate.py
```

5. Run inference on real-world images:

```bash
python scripts/predict.py --input Realworld_data/
```

## 🧠 What This Project Demonstrates

- End-to-end machine learning pipeline design
- Reproducible configuration-driven development
- Clean separation between training, evaluation, and inference
- Practical use of model checkpointing and early stopping
- Data-driven reporting with visualization outputs
- Ability to adapt to both local and Colab environments

## 📈 Outputs Generated After Training

The training pipeline saves model artifacts and reports automatically to:

- `outputs/models/waste_classifier.keras`
- `outputs/models/best_model.keras`
- `outputs/models/checkpoints/`
- `outputs/reports/training_curves.png`
- `outputs/reports/confusion_matrix.png`
- `outputs/reports/classification_report.txt`
- `outputs/reports/roc_curve.png`
- `outputs/reports/precision_recall_curve.png`

## 🧪 Important Scripts

- `scripts/train.py` — full training pipeline with optional resume support
- `scripts/evaluate.py` — standalone test-set evaluation of a saved model
- `scripts/predict.py` — inference pipeline for real-world images

## 💡 Recommended Workflow

1. Review `config.yaml` to adjust image size, batch size, augmentation, training schedule, and output paths.
2. Run `python scripts/train.py` to train and generate evaluation artifacts.
3. Use `python scripts/evaluate.py` to verify model performance on the held-out test set.
4. Use `python scripts/predict.py --input Realworld_data/` to validate inference on new images.

## 📚 Additional Resources

- `docs/training.md` — training commands and expected outputs
- `docs/inference.md` — inference workflow and result format
- `docs/GUIDE.md` — step-by-step usage guide for Colab and local runs
