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
   - Saves     : outputs/models/best_model.keras   (local)
   - Condition : only saved when val_loss improves (save_best_only=True)
   - Purpose   : always have the best-ever checkpoint available even if
                 EarlyStopping has not yet triggered

3. ModelCheckpoint — every epoch
   - Saves     : outputs/models/checkpoints/epoch_NN_valloss_X.XXXX.keras  (local)
   - Condition : saved after every epoch unconditionally
   - Purpose   : full training audit trail; allows manual inspection of any
                 intermediate checkpoint

4. DriveSync (custom)
   - Runs AFTER ModelCheckpoints at each epoch end
   - Every epoch : copies the new epoch checkpoint to Google Drive
                   (drive_sync must be true in config.yaml)
   - Best model  : copies best_model.keras to Google Drive, REPLACING the
                   previous copy — only when val_loss improved this epoch
   - Skipped silently when drive_sync=false or Drive path is unreachable

5. EpochLogger (custom)
   - Prints a rich per-epoch report:
       Training / Validation loss, Accuracy, Precision, Recall, F1,
       class-wise metrics, confusion matrix, checkpoint + Drive sync status.

Dropout regularisation
-----------------------
Dropout(0.5) is applied inside the model (see model.py).  It is active
during training and automatically disabled during evaluation/inference by
Keras — no manual intervention needed.

Public API
----------
    get_callbacks(val_data)       → list[Callback]
    train(model, train_data, val_data) → history
    save_model(model)
"""

import shutil
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
    """Create all required output directories if they don't already exist."""
    CFG.models_dir.mkdir(parents=True, exist_ok=True)
    CFG.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    CFG.reports_dir.mkdir(parents=True, exist_ok=True)
    if CFG.drive_sync:
        try:
            CFG.gdrive_checkpoints_dir.mkdir(parents=True, exist_ok=True)
            CFG.gdrive_best_model_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Drive not mounted yet — will be checked again in DriveSync.on_epoch_end
            pass


def resume_from_checkpoint() -> tuple:
    """
    Attempt to resume training from the latest epoch checkpoint stored on
    Google Drive.

    How it works
    ------------
    1. Scans CFG.gdrive_checkpoints_dir for files matching the epoch filename
       template  ``epoch_{NN}_valloss_{X}.keras``.
    2. Picks the file with the highest epoch number.
    3. Copies it back to CFG.checkpoints_dir (local Colab disk).
    4. Loads its weights into a freshly built model.
    5. Returns (model, initial_epoch) so model.fit() starts from the right
       epoch number instead of 0.

    Also restores best_model.keras from Drive if it is present there but
    missing locally (needed so ModelCheckpoint's internal best-loss tracking
    starts from the right baseline).

    Returns
    -------
    model         : tf.keras.Model with weights loaded from the checkpoint
    initial_epoch : int  — the epoch number to pass to model.fit() as
                    ``initial_epoch``; equals 0 when no checkpoint is found

    Raises
    ------
    RuntimeError  if drive_sync is False (nothing to resume from).
    """
    import re

    if not CFG.drive_sync:
        raise RuntimeError(
            "resume_from_checkpoint() requires drive_sync=true in config.yaml"
        )

    drive_ckpt_dir = CFG.gdrive_checkpoints_dir
    if not drive_ckpt_dir.exists():
        print("  [Resume] Drive checkpoints directory not found — starting fresh.")
        from waste_classifier.model import build_model
        return build_model(), 0

    # ── Find all epoch checkpoint files on Drive ─────────────────────────
    # Filename pattern: epoch_02_valloss_0.3812.keras
    pattern = re.compile(r"epoch_(\d+)_valloss_([\d.]+)\.keras$")
    candidates = []
    for f in drive_ckpt_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            candidates.append((int(m.group(1)), f))

    if not candidates:
        print("  [Resume] No epoch checkpoints found on Drive — starting fresh.")
        from waste_classifier.model import build_model
        return build_model(), 0

    # ── Pick the latest ───────────────────────────────────────────────────
    candidates.sort(key=lambda x: x[0])
    latest_epoch, latest_path = candidates[-1]

    print(f"  [Resume] Found {len(candidates)} checkpoint(s) on Drive.")
    print(f"  [Resume] Resuming from: {latest_path.name}  (epoch {latest_epoch})")

    # ── Copy from Drive → local checkpoints dir ───────────────────────────
    _ensure_output_dirs()
    local_dest = CFG.checkpoints_dir / latest_path.name
    shutil.copy2(latest_path, local_dest)
    print(f"  [Resume] Copied to local: {local_dest}")

    # ── Restore best_model.keras from Drive if missing locally ───────────
    drive_best = CFG.gdrive_best_model_dir / CFG.best_model_path.name
    if drive_best.exists() and not CFG.best_model_path.exists():
        shutil.copy2(drive_best, CFG.best_model_path)
        print(f"  [Resume] Restored best_model.keras from Drive → {CFG.best_model_path}")

    # ── Load the model ────────────────────────────────────────────────────
    print("  [Resume] Loading weights from checkpoint...")
    model = tf.keras.models.load_model(str(local_dest))
    print(f"  [Resume] Ready — will continue from epoch {latest_epoch + 1}.")

    return model, latest_epoch


# ---------------------------------------------------------------------------
# Custom callback — Google Drive sync
# ---------------------------------------------------------------------------

class DriveSync(Callback):
    """
    Copy checkpoints and the best model to Google Drive after each epoch.

    Rules
    -----
    - Every epoch  : the per-epoch .keras file (epoch_NN_valloss_X.keras) is
                     copied to CFG.gdrive_checkpoints_dir.  One file is added
                     per epoch; nothing is ever deleted from Drive.
    - Best model   : best_model.keras is copied to CFG.gdrive_best_model_dir,
                     REPLACING the previous copy.  This only happens when
                     val_loss improved (i.e. the local best_model.keras was
                     just re-written by ModelCheckpoint).

    The callback is fully self-contained — it uses shutil.copy2 (stdlib) and
    needs no extra packages.  If Drive is not mounted the copy is skipped and
    a warning is printed instead of raising.

    Parameters
    ----------
    best_val_loss_ref : list[float]
        A one-element list shared with EpochLogger so both callbacks agree on
        the current best val_loss without coupling them directly.
    """

    def __init__(self, best_val_loss_ref: list):
        super().__init__()
        self._best_val_loss_ref = best_val_loss_ref   # shared [float("inf")]

    def on_epoch_end(self, epoch: int, logs: dict = None):
        if not CFG.drive_sync:
            return

        logs      = logs or {}
        val_loss  = logs.get("val_loss", float("inf"))
        epoch_num = epoch + 1

        # ── Resolve the local epoch-checkpoint filename ──────────────────
        # The template in config is:  epoch_{epoch:02d}_valloss_{val_loss:.4f}.keras
        # Keras uses 1-based epoch in the filename when save_freq="epoch".
        epoch_filename = CFG.ckpt_epoch_template.format(
            epoch    = epoch_num,
            val_loss = val_loss,
        )
        local_epoch_ckpt = CFG.checkpoints_dir / epoch_filename

        # ── Copy epoch checkpoint to Drive ───────────────────────────────
        try:
            CFG.gdrive_checkpoints_dir.mkdir(parents=True, exist_ok=True)
            if local_epoch_ckpt.exists():
                shutil.copy2(local_epoch_ckpt, CFG.gdrive_checkpoints_dir / epoch_filename)
                self._epoch_synced = True
            else:
                print(f"\n  [DriveSync] WARNING: local checkpoint not found: {local_epoch_ckpt}")
                self._epoch_synced = False
        except OSError as exc:
            print(f"\n  [DriveSync] WARNING: could not copy epoch checkpoint to Drive: {exc}")
            self._epoch_synced = False

        # ── Copy best model to Drive (only when val_loss improved) ───────
        is_new_best = val_loss < self._best_val_loss_ref[0]
        if is_new_best:
            self._best_val_loss_ref[0] = val_loss
            try:
                CFG.gdrive_best_model_dir.mkdir(parents=True, exist_ok=True)
                dest = CFG.gdrive_best_model_dir / CFG.best_model_path.name
                shutil.copy2(CFG.best_model_path, dest)
                self._best_synced = True
            except OSError as exc:
                print(f"\n  [DriveSync] WARNING: could not copy best model to Drive: {exc}")
                self._best_synced = False
        else:
            self._best_synced = False

        # Store for EpochLogger to read
        self._last_is_new_best = is_new_best


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
    - Checkpoint & Drive Sync status

    Parameters
    ----------
    val_data : Keras DirectoryIterator
        Validation generator — reset and iterated once per epoch to collect
        true labels and predictions.
    class_names : list[str]
        Ordered class labels matching the generator's class_indices order.
    best_model_path : Path
        Path where ModelCheckpoint saves the best model (used for display).
    drive_sync_cb : DriveSync or None
        Reference to the DriveSync callback so Drive status can be read and
        shown in the report.  Pass None when drive_sync=false.
    """

    def __init__(
        self,
        val_data,
        class_names: list,
        best_model_path: Path,
        drive_sync_cb=None,
    ):
        super().__init__()
        self.val_data        = val_data
        self.class_names     = class_names
        self.best_model_path = best_model_path
        self.drive_sync_cb   = drive_sync_cb
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

        # ── Drive sync status (read from DriveSync callback) ─────────────
        ds = self.drive_sync_cb
        if ds is not None and CFG.drive_sync:
            epoch_synced = getattr(ds, "_epoch_synced", False)
            best_synced  = getattr(ds, "_best_synced",  False)
        else:
            epoch_synced = None   # None = Drive sync disabled
            best_synced  = None

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
        print(f"    Local — Best Model Saved  : {'YES' if saved else 'NO'}")
        if self._best_epoch is not None:
            print(f"    Local — Best Epoch        : {self._best_epoch}")
        print(f"    Local — Model Path        : {self.best_model_path}")

        if epoch_synced is None:
            print(f"\n    Drive Sync                : disabled")
        else:
            ckpt_status = "OK" if epoch_synced else "FAILED"
            best_status = ("OK" if best_synced else "FAILED") if saved else "skipped (not best)"
            print(f"\n    Drive — Epoch Checkpoint  : {ckpt_status}")
            print(f"    Drive — Best Model        : {best_status}")
            print(f"    Drive — Checkpoints Dir   : {CFG.gdrive_checkpoints_dir}")
            print(f"    Drive — Best Model Dir    : {CFG.gdrive_best_model_dir}")

        print("\n" + sep2 + "\n")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def get_callbacks(val_data=None) -> list:
    """
    Build and return the five training callbacks.

    Callback execution order within each epoch
    -------------------------------------------
    Keras calls on_epoch_end in list order.  The ordering here guarantees:
      1. EarlyStopping  — decides whether to stop (reads val_loss)
      2. ModelCheckpoint best  — writes best_model.keras if val_loss improved
      3. ModelCheckpoint epoch — writes epoch_NN_valloss_X.keras unconditionally
      4. DriveSync      — copies the freshly-written local files to Drive
      5. EpochLogger    — reads Drive status from DriveSync, prints full report

    Parameters
    ----------
    val_data : Keras DirectoryIterator, optional
        Required by EpochLogger and DriveSync.
        If None, both are omitted from the returned list.

    Returns
    -------
    list of Keras Callbacks in execution order.
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
        filepath          = str(CFG.best_model_path),   # outputs/models/best_model.keras
        monitor           = CFG.ckpt_monitor,           # "val_loss"
        save_best_only    = True,
        save_weights_only = False,                      # Save full model, not just weights
        mode              = "min",                      # Lower val_loss = better
        verbose           = 1,
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
        # Shared mutable reference so DriveSync and EpochLogger track the same
        # best val_loss without either depending on the other's internals.
        best_val_loss_ref = [float("inf")]

        # ── 4. DriveSync ─────────────────────────────────────────────────
        drive_sync_cb = DriveSync(best_val_loss_ref=best_val_loss_ref)
        callbacks.append(drive_sync_cb)

        # ── 5. EpochLogger ───────────────────────────────────────────────
        epoch_logger = EpochLogger(
            val_data        = val_data,
            class_names     = CFG.class_names,
            best_model_path = CFG.best_model_path,
            drive_sync_cb   = drive_sync_cb if CFG.drive_sync else None,
        )
        callbacks.append(epoch_logger)

    return callbacks


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: tf.keras.Model,
    train_data,
    val_data,
    initial_epoch: int = 0,
) -> tf.keras.callbacks.History:
    """
    Run the training loop with all callbacks.

    Parameters
    ----------
    model         : compiled tf.keras.Model
    train_data    : training DirectoryIterator
    val_data      : validation DirectoryIterator
    initial_epoch : int, default 0
                    Epoch to start from.  Pass the value returned by
                    resume_from_checkpoint() to continue an interrupted run.
                    Keras will count from this number up to CFG.epochs, so
                    effectively only (CFG.epochs - initial_epoch) epochs run.

    Returns
    -------
    history : tf.keras.callbacks.History
    """
    remaining = CFG.epochs - initial_epoch

    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)
    print(f"  Max epochs      : {CFG.epochs}")
    if initial_epoch > 0:
        print(f"  Resuming from   : epoch {initial_epoch + 1}  ({remaining} epoch(s) remaining)")
    print(f"  EarlyStopping   : patience={CFG.es_patience}, monitor={CFG.es_monitor}")
    print(f"  Best checkpoint : {CFG.best_model_path.name}")
    print(f"  Epoch saves     : {CFG.checkpoints_dir}")
    print(f"  Dropout rate    : {CFG.dropout_rate}  (active during training only)")
    print(f"  Drive sync      : {'ON  -> ' + str(CFG.gdrive_checkpoints_dir) if CFG.drive_sync else 'OFF'}")
    print("=" * 60 + "\n")

    model.summary()

    history = model.fit(
        train_data,
        validation_data = val_data,
        epochs          = CFG.epochs,
        initial_epoch   = initial_epoch,
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
