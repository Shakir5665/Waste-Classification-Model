# Training Guide

## Prerequisites

- Python 3.8+
- GPU recommended (CUDA + cuDNN, or Google Colab T4)
- Dataset structured as `dataset/Train/`, `dataset/Validation/`, `dataset/Test/`

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python scripts/train.py
```

This single command runs the full pipeline:
1. Loads all settings from `config.yaml`
2. Creates augmented data generators
3. Builds and compiles the CNN
4. Trains with EarlyStopping (patience=3 on `val_loss`)
5. Saves model → `outputs/models/waste_classifier.keras`
6. Saves training curves → `outputs/reports/training_curves.png`
7. Saves confusion matrix → `outputs/reports/confusion_matrix.png`
8. Saves classification report → `outputs/reports/classification_report.txt`

## Training Configuration

All training hyperparameters are controlled via `config.yaml`:

```yaml
training:
  epochs: 20          # Maximum epochs (early stopping may stop sooner)
  optimizer: "adam"
  loss: "binary_crossentropy"
  metrics: ["accuracy"]

callbacks:
  early_stopping:
    monitor: "val_loss"
    patience: 3
    restore_best_weights: true
```

## Expected Output

```
[1/4] Loading dataset...
Found 4462 images belonging to 2 classes.
Found 900 images belonging to 2 classes.
Found 1680 images belonging to 2 classes.

[2/4] Building model...

[3/4] Training...
Epoch 1/20  ...
...
Epoch N/20  ... (EarlyStopping restores best weights)

[4/4] Evaluating...
  Test Loss     : 0.1412
  Test Accuracy : 95.30%
```

## Google Colab

Open `notebooks/colab_runner.ipynb` in Colab and run each cell in order.
The notebook clones the repository, installs dependencies, uploads the dataset,
and calls `scripts/train.py` — it contains no logic of its own.

## Standalone Evaluation (Re-evaluate a Saved Model)

```bash
python scripts/evaluate.py
# or with a custom model path:
python scripts/evaluate.py --model outputs/models/waste_classifier.keras
```
