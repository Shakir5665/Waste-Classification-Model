"""
evaluate.py
-----------
Evaluation framework — every industry-standard metric, plot, and report.

Graphs and reports produced
----------------------------
  1.  Training curves         — accuracy + loss (train vs. validation) per epoch
  2.  Confusion matrix        — absolute counts, with class names
  3.  Confusion matrix (norm) — row-normalised to percentages (shows per-class rates)
  4.  ROC curve               — AUC score printed on the plot
  5.  Precision-Recall curve  — Average Precision (AP) printed on the plot
  6.  Per-class metrics bar   — Precision / Recall / F1 per class as grouped bars
  7.  Classification report   — full sklearn report saved as .txt
  8.  Training summary JSON   — all scalar metrics in one machine-readable file

Techniques confirmed present
-----------------------------
  - Dropout(0.5)       → defined in model.py; active during training,
                          auto-disabled during evaluation by Keras
  - EarlyStopping      → defined in train.py callbacks
  - ModelCheckpoint    → best_model.keras + per-epoch checkpoints (train.py)

Public API
----------
    evaluate_model(model, test_data)                    → (loss, acc)
    plot_training_curves(history, save=True)            → None
    plot_confusion_matrix(model, test_data, save=True)  → np.ndarray  (cm)
    plot_roc_curve(y_true, y_prob, save=True)           → float (auc)
    plot_pr_curve(y_true, y_prob, save=True)            → float (ap)
    plot_per_class_metrics(y_true, y_pred, save=True)   → None
    print_classification_report(y_true, y_pred, save=True) → str
    save_training_summary(history, test_loss, test_acc) → None
    run_full_evaluation(model, history, test_data)      → None   ← main entry point
"""

import json
import numpy as np

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — safe for scripts & Colab
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

import tensorflow as tf

from waste_classifier.config import CFG


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_reports_dir() -> None:
    CFG.reports_dir.mkdir(parents=True, exist_ok=True)


def _get_predictions(model: tf.keras.Model, test_data):
    """
    Run inference once on the entire test set and return
    (true_classes, predicted_classes, raw_probabilities).

    Reuses a single model.predict() call to avoid running inference twice.
    """
    y_prob  = model.predict(test_data, verbose=1).flatten()   # sigmoid output [0,1]
    y_pred  = (y_prob >= CFG.threshold).astype(int)           # binary labels
    y_true  = test_data.classes                               # ground truth labels
    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# 1. Test-set evaluation (loss + accuracy)
# ---------------------------------------------------------------------------

def evaluate_model(model: tf.keras.Model, test_data) -> tuple:
    """
    Evaluate the model on the test set using model.evaluate().

    Returns
    -------
    (test_loss, test_accuracy) : tuple[float, float]
    """
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)
    test_loss, test_acc = model.evaluate(test_data, verbose=1)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    return test_loss, test_acc


# ---------------------------------------------------------------------------
# 2. Training curves — accuracy + loss, side-by-side
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: tf.keras.callbacks.History,
    save: bool = True,
) -> None:
    """
    Plot training vs. validation accuracy AND loss in one figure with
    two side-by-side subplots.

    - Left  : Accuracy (train + validation)
    - Right : Loss     (train + validation)

    A vertical dashed line marks the best epoch (lowest val_loss).
    """
    h = history.history
    epochs_ran = range(1, len(h["accuracy"]) + 1)
    best_epoch = int(np.argmin(h["val_loss"])) + 1   # 1-based

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History", fontsize=15, fontweight="bold")

    # ── Accuracy ────────────────────────────────────────────────────────────
    ax1.plot(epochs_ran, h["accuracy"],     label="Train",      marker="o", linewidth=2)
    ax1.plot(epochs_ran, h["val_accuracy"], label="Validation", marker="o", linewidth=2)
    ax1.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                label=f"Best epoch ({best_epoch})")
    ax1.set_title("Model Accuracy", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_ylim([0, 1.05])

    # ── Loss ────────────────────────────────────────────────────────────────
    ax2.plot(epochs_ran, h["loss"],     label="Train",      marker="o", linewidth=2)
    ax2.plot(epochs_ran, h["val_loss"], label="Validation", marker="o", linewidth=2)
    ax2.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                label=f"Best epoch ({best_epoch})")
    ax2.set_title("Model Loss", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save:
        _ensure_reports_dir()
        fig.savefig(str(CFG.curves_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {CFG.curves_path.name}")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Confusion matrix — absolute counts
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save: bool = True,
) -> np.ndarray:
    """
    Plot the confusion matrix with actual class names (Organic / Recyclable).

    Also plots a row-normalised version (percentages) as a second figure.

    Returns
    -------
    cm : np.ndarray  shape (2, 2)
    """
    cm = confusion_matrix(y_true, y_pred)

    # ── Absolute counts ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = "d",
        cmap        = "Blues",
        xticklabels = CFG.class_names,
        yticklabels = CFG.class_names,
        ax          = ax,
        linewidths  = 0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        _ensure_reports_dir()
        fig.savefig(str(CFG.cm_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {CFG.cm_path.name}")

    plt.show()
    plt.close(fig)

    # ── Row-normalised (percentages) ─────────────────────────────────────────
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot       = True,
        fmt         = ".2%",
        cmap        = "Greens",
        xticklabels = CFG.class_names,
        yticklabels = CFG.class_names,
        ax          = ax2,
        linewidths  = 0.5,
        vmin        = 0,
        vmax        = 1,
    )
    ax2.set_xlabel("Predicted Label", fontsize=11)
    ax2.set_ylabel("True Label", fontsize=11)
    ax2.set_title("Confusion Matrix (Normalised %)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        fig2.savefig(str(CFG.cm_norm_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {CFG.cm_norm_path.name}")

    plt.show()
    plt.close(fig2)

    return cm


# ---------------------------------------------------------------------------
# 4. ROC curve + AUC
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save: bool = True,
) -> float:
    """
    Plot the ROC (Receiver Operating Characteristic) curve.
    Prints and returns the AUC (Area Under Curve) score.

    A model that classifies randomly has AUC = 0.50.
    A perfect model has AUC = 1.00.

    Returns
    -------
    roc_auc : float
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
            label="Random Classifier (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        _ensure_reports_dir()
        fig.savefig(str(CFG.roc_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {CFG.roc_path.name}  (AUC = {roc_auc:.4f})")

    plt.show()
    plt.close(fig)
    return roc_auc


# ---------------------------------------------------------------------------
# 5. Precision-Recall curve + Average Precision
# ---------------------------------------------------------------------------

def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save: bool = True,
) -> float:
    """
    Plot the Precision-Recall curve.
    Prints and returns the Average Precision (AP) score.

    AP summarises the curve as the weighted mean of precisions at each
    threshold.  More informative than ROC when classes are imbalanced.

    Returns
    -------
    avg_precision : float
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()   # fraction of positives — random classifier baseline

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="darkorange", linewidth=2,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Random Classifier (AP ≈ {baseline:.2f})")
    ax.fill_between(recall, precision, alpha=0.1, color="darkorange")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        _ensure_reports_dir()
        fig.savefig(str(CFG.pr_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {CFG.pr_path.name}  (AP = {ap:.4f})")

    plt.show()
    plt.close(fig)
    return ap


# ---------------------------------------------------------------------------
# 6. Per-class metrics bar chart
# ---------------------------------------------------------------------------

def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save: bool = True,
) -> None:
    """
    Grouped bar chart showing Precision, Recall, and F1-score for each class.

    Gives a clear visual of where the model is strong or weak on a
    per-class basis.
    """
    metrics_per_class = {}
    for i, class_name in enumerate(CFG.class_names):
        # binarise to 1 = this class, 0 = other
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        metrics_per_class[class_name] = {
            "Precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
            "Recall"   : recall_score(y_true_bin, y_pred_bin, zero_division=0),
            "F1-Score" : f1_score(y_true_bin, y_pred_bin, zero_division=0),
        }

    metric_names = ["Precision", "Recall", "F1-Score"]
    x = np.arange(len(metric_names))
    bar_width = 0.30
    colors = ["steelblue", "darkorange"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (class_name, scores) in enumerate(metrics_per_class.items()):
        values = [scores[m] for m in metric_names]
        offset = (idx - (len(CFG.class_names) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                      label=class_name, color=colors[idx], alpha=0.85)
        # Label each bar with its value
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.set_title("Per-Class Precision, Recall & F1-Score", fontsize=13, fontweight="bold")
    ax.legend(title="Class", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        _ensure_reports_dir()
        fig.savefig(str(CFG.metrics_bar_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {CFG.metrics_bar_path.name}")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Classification report (text)
# ---------------------------------------------------------------------------

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save: bool = True,
) -> str:
    """
    Print and save the full sklearn classification report.

    Returns
    -------
    report_str : str
    """
    report_str = classification_report(
        y_true,
        y_pred,
        target_names = CFG.class_names,
        digits       = 4,
    )

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_str)

    if save:
        _ensure_reports_dir()
        with open(str(CFG.report_path), "w") as fh:
            fh.write(report_str)
        print(f"  Saved: {CFG.report_path.name}")

    return report_str


# ---------------------------------------------------------------------------
# 8. Training summary JSON
# ---------------------------------------------------------------------------

def save_training_summary(
    history: tf.keras.callbacks.History,
    test_loss: float,
    test_acc: float,
    roc_auc: float,
    avg_precision: float,
) -> None:
    """
    Write a machine-readable JSON summary of the complete training run.

    Includes:
      - best epoch and its val_loss / val_accuracy
      - total epochs trained
      - final test_loss and test_accuracy
      - ROC AUC and Average Precision
      - full per-epoch history arrays

    Saved to outputs/reports/training_summary.json
    """
    h = history.history
    best_epoch = int(np.argmin(h["val_loss"]))

    summary = {
        "training": {
            "total_epochs_run" : len(h["loss"]),
            "best_epoch"       : best_epoch + 1,          # 1-based
            "best_val_loss"    : float(h["val_loss"][best_epoch]),
            "best_val_accuracy": float(h["val_accuracy"][best_epoch]),
        },
        "test_set": {
            "test_loss"        : float(test_loss),
            "test_accuracy"    : float(test_acc),
            "roc_auc"          : float(roc_auc),
            "average_precision": float(avg_precision),
        },
        "history": {
            "accuracy"    : [float(v) for v in h["accuracy"]],
            "val_accuracy": [float(v) for v in h["val_accuracy"]],
            "loss"        : [float(v) for v in h["loss"]],
            "val_loss"    : [float(v) for v in h["val_loss"]],
        },
        "config": {
            "img_size"    : list(CFG.img_size),
            "batch_size"  : CFG.batch_size,
            "epochs_max"  : CFG.epochs,
            "dropout_rate": CFG.dropout_rate,
            "threshold"   : CFG.threshold,
        },
    }

    _ensure_reports_dir()
    with open(str(CFG.summary_path), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved: {CFG.summary_path.name}")


# ---------------------------------------------------------------------------
# Main entry point — runs the complete evaluation suite
# ---------------------------------------------------------------------------

def run_full_evaluation(
    model: tf.keras.Model,
    history,           # tf.keras.callbacks.History or None (standalone eval)
    test_data,
) -> None:
    """
    Run the complete evaluation suite in one call.

    Steps
    -----
    1.  model.evaluate()         → test loss + accuracy
    2.  model.predict()          → raw probabilities (single inference pass)
    3.  Training curves          → accuracy + loss plots  (skipped if history=None)
    4.  Confusion matrix         → absolute counts
    5.  Confusion matrix (norm)  → row-normalised percentages
    6.  ROC curve                → with AUC score
    7.  Precision-Recall curve   → with Average Precision
    8.  Per-class metrics bar    → Precision / Recall / F1 grouped bars
    9.  Classification report    → saved as .txt
    10. Training summary JSON    → all scalars in one file  (skipped if history=None)

    All artefacts saved to outputs/reports/
    """
    print("\n" + "=" * 60)
    print("  FULL EVALUATION SUITE")
    print("=" * 60)

    # ── Step 1: test loss + accuracy ────────────────────────────────────────
    test_loss, test_acc = evaluate_model(model, test_data)

    # ── Step 2: single inference pass ───────────────────────────────────────
    print("\n  Running predictions on test set...")
    y_true, y_pred, y_prob = _get_predictions(model, test_data)

    # ── Step 3: training curves ──────────────────────────────────────────────
    print("\n[Graph 1/6] Training curves (accuracy + loss)...")
    if history is not None:
        plot_training_curves(history, save=True)
    else:
        print("  Skipped — no training history available.")

    # ── Steps 4-5: confusion matrices ────────────────────────────────────────
    print("\n[Graph 2/6] Confusion matrix (counts + normalised)...")
    plot_confusion_matrix(y_true, y_pred, save=True)

    # ── Step 6: ROC curve ────────────────────────────────────────────────────
    print("\n[Graph 3/6] ROC curve...")
    roc_auc = plot_roc_curve(y_true, y_prob, save=True)

    # ── Step 7: Precision-Recall curve ───────────────────────────────────────
    print("\n[Graph 4/6] Precision-Recall curve...")
    avg_precision = plot_pr_curve(y_true, y_prob, save=True)

    # ── Step 8: per-class metrics bar ────────────────────────────────────────
    print("\n[Graph 5/6] Per-class metrics bar chart...")
    plot_per_class_metrics(y_true, y_pred, save=True)

    # ── Step 9: classification report ────────────────────────────────────────
    print("\n[Graph 6/6] Classification report...")
    print_classification_report(y_true, y_pred, save=True)

    # ── Step 10: training summary JSON ───────────────────────────────────────
    if history is not None:
        print("\n  Training summary JSON...")
        save_training_summary(history, test_loss, test_acc, roc_auc, avg_precision)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Test Accuracy    : {test_acc * 100:.2f}%")
    print(f"  Test Loss        : {test_loss:.4f}")
    print(f"  ROC AUC          : {roc_auc:.4f}")
    print(f"  Avg Precision    : {avg_precision:.4f}")
    print(f"\n  All reports saved to: {CFG.reports_dir}")
    print("=" * 60)
