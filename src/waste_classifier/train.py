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

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
# Callbacks
# ---------------------------------------------------------------------------

def get_callbacks() -> list:
    """
    Build and return the three training callbacks.

    Returns
    -------
    list containing:
        [0] EarlyStopping         — stops training when val_loss stagnates
        [1] ModelCheckpoint best  — saves outputs/models/best_model.keras
        [2] ModelCheckpoint epoch — saves outputs/models/checkpoints/epoch_NN_...keras
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

    return [early_stop, best_ckpt, epoch_ckpt]


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
        callbacks       = get_callbacks(),
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
