"""
train.py
--------
Training pipeline.

Callbacks used during training
-------------------------------
1. EarlyStopping
   - Monitors : val_loss
   - Patience  : 3 epochs  (stops when val_loss does not improve for 3 epochs)
   - Action    : restores the best weights automatically

2. ModelCheckpoint — best model only
   - Saves     : outputs/models/best_model.keras
   - Condition : only saved when val_loss improves (save_best_only=True)
   - Purpose   : always have the best-ever checkpoint available even if
                 EarlyStopping has not yet triggered

3. ModelCheckpoint — every epoch
   - Saves     : outputs/models/checkpoints/epoch_NN_valloss_X.XXXX.keras
   - Condition : saved after every epoch unconditionally
   - Purpose   : full training audit trail; allows manual inspection of any
                 intermediate checkpoint

4. EpochLogger (custom)
   - Prints a rich per-epoch report after each epoch:
       Training / Validation loss, Accuracy, Precision, Recall, F1,
       class-wise metrics, confusion matrix, and checkpoint status.

Dropout regularisation
-----------------------
Dropout(0.5) is applied inside the model (see model.py).  It is active
during training and automatically disabled during evaluation/inference by
Keras — no manual intervention needed.

Public API
----------
    get_callbacks()               → list[Callback]
    train(model, train_data, val_data) → history
    save_model(model)
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_fscore_support,
)

from waste_classifier.config import CFG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_output_dirs() -> None:
    """Create all output directories if they don't already exist."""
    CFG.models_dir.mkdir(parents=True, exist_ok=True)
    CFG.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    CFG.reports_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Custom callback — rich per-epoch report
# ---------------------------------------------------------------------------

class EpochLogger(Callback):
    """
    Print a detailed, human-readable report at the end of every epoch.

    Report sections
    ---------------
    - Training Loss   (cross-entropy reported by model.fit)
    - Validation Loss (cross-entropy reported by model.fit)
    - Classification Metrics  (accuracy, precision, recall, F1) — macro avg
    - Class-wise Performance  (precision, recall, F1 per class)
    - Confusion Matrix        (raw counts)
    - Best Model Saved        (YES / NO, epoch number, path)

    Parameters
    ----------
    val_data : Keras DirectoryIterator
        Validation generator — reset and iterated once per epoch to collect
        true labels and predictions.
    class_names : list[str]
        Ordered class labels matching the generator's class_indices order.
    best_model_path : Path
        Path where ModelCheckpoint saves the best model (used for display).
    """

    def __init__(self, val_data, class_names: list, best_model_path: Path):
        super().__init__()
        self.val_data        = val_data
        self.class_names     = class_names
        self.best_model_path = best_model_path
        self._best_val_loss  = float("inf")
        self._best_epoch     = None

    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        epoch_num = epoch + 1  # 1-based

        # ── Collect predictions over the full validation set ────────────
        self.val_data.reset()
        y_true, y_prob = [], []
        for _ in range(len(self.val_data)):
            x_batch, y_batch = next(self.val_data)
            probs = self.model.predict(x_batch, verbose=0).flatten()
            y_prob.extend(probs.tolist())
            y_true.extend(y_batch.tolist())

        y_true  = np.array(y_true, dtype=int)
        y_prob  = np.array(y_prob)
        y_pred  = (y_prob >= CFG.threshold).astype(int)

        # ── Scalar metrics ───────────────────────────────────────────────
        train_loss = logs.get("loss",     float("nan"))
        val_loss   = logs.get("val_loss", float("nan"))
        accuracy   = float(np.mean(y_true == y_pred))

        precision_macro = precision_score(y_true, y_pred, average="macro",  zero_division=0)
        recall_macro    = recall_score   (y_true, y_pred, average="macro",  zero_division=0)
        f1_macro        = f1_score       (y_true, y_pred, average="macro",  zero_division=0)

        # Per-class: returns arrays ordered by class index (0, 1, ...)
        p_per, r_per, f_per, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0,
        )

        # ── Confusion matrix ─────────────────────────────────────────────
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # ── Best-model tracking ──────────────────────────────────────────
        saved = val_loss < self._best_val_loss
        if saved:
            self._best_val_loss = val_loss
            self._best_epoch    = epoch_num

        # ── Column widths for confusion matrix ───────────────────────────
        c0, c1 = self.class_names[0].capitalize(), self.class_names[1].capitalize()
        col_w  = max(len(c0), len(c1), 6) + 2   # minimum 8 chars

        # -- Print report --------------------------------------------------
        sep  = "-" * 52
        sep2 = "=" * 52

        print("\n" + sep2)
        print(f"  Epoch {epoch_num:>3} / {self.params.get('epochs', '?'):<3}  Summary")
        print(sep2 + "\n")

        print("  Training Loss\n")
        print(f"    Cross-Entropy : {train_loss:.4f}")
        print(f"    Total Loss    : {train_loss:.4f}")

        print("\n  " + sep)
        print("  Validation Loss\n")
        print(f"    Cross-Entropy : {val_loss:.4f}")
        print(f"    Total Loss    : {val_loss:.4f}")

        print("\n  " + sep)
        print("  Classification Metrics\n")
        print(f"    Accuracy  : {accuracy:.4f}")
        print(f"    Precision : {precision_macro:.4f}")
        print(f"    Recall    : {recall_macro:.4f}")
        print(f"    F1 Score  : {f1_macro:.4f}")

        print("\n  " + sep)
        print("  Class-wise Performance\n")
        for idx, name in enumerate(self.class_names):
            print(f"    {name.capitalize()}:")
            print(f"      Precision : {p_per[idx]:.4f}")
            print(f"      Recall    : {r_per[idx]:.4f}")
            print(f"      F1 Score  : {f_per[idx]:.4f}")
            print()

        print("  " + sep)
        print("  Confusion Matrix\n")
        print("                    Predicted")
        print(f"                    {c0:<{col_w}}{c1}")
        print("    Actual")
        print(f"    {c0:<18}  {cm[0,0]:<{col_w}}{cm[0,1]}")
        print(f"    {c1:<18}  {cm[1,0]:<{col_w}}{cm[1,1]}")

        print("\n  " + sep)
        print("  Checkpoint\n")
        print(f"    Best Model Saved : {'YES' if saved else 'NO'}")
        if self._best_epoch is not None:
            print(f"    Best Epoch       : {self._best_epoch}")
        print(f"    Model Path       : {self.best_model_path}")
        print("\n" + sep2 + "\n")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def get_callbacks(val_data=None) -> list:
    """
    Build and return the four training callbacks.

    Parameters
    ----------
    val_data : Keras DirectoryIterator, optional
        Required by EpochLogger to compute per-epoch validation metrics.
        If None, EpochLogger is omitted from the returned list.

    Returns
    -------
    list containing:
        [0] EarlyStopping         — stops training when val_loss stagnates
        [1] ModelCheckpoint best  — saves outputs/models/best_model.keras
        [2] ModelCheckpoint epoch — saves outputs/models/checkpoints/epoch_NN_...keras
        [3] EpochLogger           — prints rich per-epoch report (when val_data given)
    """
    _ensure_output_dirs()

    # ── 1. EarlyStopping ────────────────────────────────────────────────────
    early_stop = EarlyStopping(
        monitor              = CFG.es_monitor,       # "val_loss"
        patience             = CFG.es_patience,      # 3
        restore_best_weights = CFG.es_restore,       # True
        verbose              = 1,
    )

    # ── 2. Best-model checkpoint ─────────────────────────────────────────────
    # Saved to outputs/models/best_model.keras only when val_loss improves.
    # This is independent of EarlyStopping — it captures the best weights
    # at any point in training as a standalone, loadable file.
    best_ckpt = ModelCheckpoint(
        filepath        = str(CFG.best_model_path),   # outputs/models/best_model.keras
        monitor         = CFG.ckpt_monitor,           # "val_loss"
        save_best_only  = True,
        save_weights_only = False,                    # Save full model, not just weights
        mode            = "min",                      # Lower val_loss = better
        verbose         = 1,
    )

    # ── 3. Every-epoch checkpoint ────────────────────────────────────────────
    # Saved to outputs/models/checkpoints/epoch_NN_valloss_X.XXXX.keras
    # after every epoch, unconditionally.  Allows full audit trail.
    epoch_ckpt_path = str(CFG.checkpoints_dir / CFG.ckpt_epoch_template)
    epoch_ckpt = ModelCheckpoint(
        filepath          = epoch_ckpt_path,
        monitor           = CFG.ckpt_monitor,         # "val_loss"
        save_best_only    = False,                    # Save every epoch
        save_weights_only = False,
        mode              = "min",
        verbose           = 0,                        # Silent — avoids log spam
    )

    callbacks = [early_stop, best_ckpt, epoch_ckpt]

    if val_data is not None:
        epoch_logger = EpochLogger(
            val_data        = val_data,
            class_names     = CFG.class_names,
            best_model_path = CFG.best_model_path,
        )
        callbacks.append(epoch_logger)

    return callbacks


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: tf.keras.Model, train_data, val_data) -> tf.keras.callbacks.History:
    """
    Run the training loop with EarlyStopping + both ModelCheckpoints.

    Parameters
    ----------
    model      : compiled tf.keras.Model  (from model.build_model())
    train_data : training DirectoryIterator  (from data.get_data_flows())
    val_data   : validation DirectoryIterator

    Returns
    -------
    history : tf.keras.callbacks.History
        Keys: 'accuracy', 'val_accuracy', 'loss', 'val_loss'  (per epoch)
    """
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)
    print(f"  Max epochs      : {CFG.epochs}")
    print(f"  EarlyStopping   : patience={CFG.es_patience}, monitor={CFG.es_monitor}")
    print(f"  Best checkpoint : {CFG.best_model_path.name}")
    print(f"  Epoch saves     : {CFG.checkpoints_dir}")
    print(f"  Dropout rate    : {CFG.dropout_rate}  (active during training only)")
    print("=" * 60 + "\n")

    model.summary()

    history = model.fit(
        train_data,
        validation_data = val_data,
        epochs          = CFG.epochs,
        callbacks       = get_callbacks(val_data),
    )
    return history


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------

def save_model(model: tf.keras.Model) -> None:
    """
    Save the final trained model (post-EarlyStopping) to
    outputs/models/waste_classifier.keras.

    Note: best_model.keras is already saved by the ModelCheckpoint callback
    during training.  This function saves the final state as a separately
    named file for clarity.

    Parameters
    ----------
    model : tf.keras.Model  (trained, best weights already restored by EarlyStopping)
    """
    _ensure_output_dirs()
    model.save(str(CFG.model_path))
    print(f"\n  Final model saved  → {CFG.model_path}")
    print(f"  Best model saved   → {CFG.best_model_path}  (saved during training)")
    print(f"  Epoch checkpoints  → {CFG.checkpoints_dir}/")
