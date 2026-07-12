"""
scripts/train.py
----------------
Entry point for the full training pipeline.

Usage
-----
    # From the repository root:
    python scripts/train.py

Colab usage
-----------
    !python scripts/train.py

What it does
------------
  1.  Loads all configuration from config.yaml
  2.  Builds data generators and directory flows
  3.  Builds and compiles the CNN (with Dropout(0.5))
  4.  Trains with three callbacks:
        - EarlyStopping (patience=3, monitors val_loss)
        - ModelCheckpoint — best_model.keras (saved when val_loss improves)
        - ModelCheckpoint — epoch_NN_valloss_X.keras (saved every epoch)
  5.  Saves final model → outputs/models/waste_classifier.keras
  6.  Runs the full evaluation suite:
        - Training curves (accuracy + loss)
        - Confusion matrix (counts)
        - Confusion matrix (normalised %)
        - ROC curve (AUC)
        - Precision-Recall curve (Average Precision)
        - Per-class metrics bar chart
        - Classification report (.txt)
        - Training summary (.json)
"""

import sys
from pathlib import Path

# ── Make the package importable when running as a script ───────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waste_classifier.config import CFG
from waste_classifier.data import get_data_flows
from waste_classifier.model import build_model
from waste_classifier import train as training_module
from waste_classifier.evaluate import run_full_evaluation


def main() -> None:
    print("=" * 60)
    print("  WASTE CLASSIFICATION MODEL — TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Dataset    : {CFG.dataset_dir}")
    print(f"  Image size : {CFG.img_size}")
    print(f"  Batch size : {CFG.batch_size}")
    print(f"  Max epochs : {CFG.epochs}")
    print(f"  Output dir : {CFG.outputs_dir}")
    print("=" * 60)

    # ── 1. Data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset...")
    train_data, val_data, test_data = get_data_flows()

    # ── 2. Model ─────────────────────────────────────────────────────────
    print("\n[2/4] Building model...")
    model = build_model()

    # ── 3. Train ─────────────────────────────────────────────────────────
    print("\n[3/4] Training...")
    print("      Callbacks: EarlyStopping + ModelCheckpoint (best) + ModelCheckpoint (every epoch)")
    history = training_module.train(model, train_data, val_data)
    training_module.save_model(model)

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    print("\n[4/4] Evaluating...")
    print("      Graphs: training curves, confusion matrix (x2), ROC, PR curve,")
    print("              per-class metrics bar, classification report, summary JSON")
    run_full_evaluation(model, history, test_data)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Final model        : {CFG.model_path}")
    print(f"  Best model         : {CFG.best_model_path}")
    print(f"  Epoch checkpoints  : {CFG.checkpoints_dir}/")
    print(f"  Reports            : {CFG.reports_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
