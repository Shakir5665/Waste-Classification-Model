"""
predict.py
----------
Inference pipeline for real-world waste image classification.

Preserves the original notebook inference logic exactly:
  - Load image → resize to 128×128 → normalise to [0,1]
  - probability >= threshold (0.5) → RECYCLABLE (green)
  - probability <  threshold       → ORGANIC    (red)
  - Displays a grid of images with labels and confidence scores
  - Prints a summary table

Improvements over the original notebook:
  - Accepts a plain directory path instead of requiring a ZIP upload widget
  - Grid visualisation is saved to outputs/reports/ as a PNG
  - Works non-interactively (no Colab-specific APIs)

Public API
----------
    predict_image(model, img_path)      → dict  {"file", "result", "confidence"}
    predict_from_directory(model, dir)  → list[dict]
    visualise_predictions(results, image_paths, save=True)
    run_inference(model, input_dir)     → list[dict]   ← convenience wrapper
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

from waste_classifier.config import CFG

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Image-level helpers
# ---------------------------------------------------------------------------

def _find_images(directory: Path) -> list:
    """Recursively find all image files in a directory."""
    found = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in _IMG_EXTENSIONS:
                found.append(Path(root) / fname)
    return found


def _preprocess_image(img_path: Path) -> np.ndarray:
    """
    Load an image, resize to model input size, and normalise to [0, 1].
    Returns array of shape (1, H, W, 3).
    """
    img = keras_image.load_img(str(img_path), target_size=CFG.img_size)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


# ---------------------------------------------------------------------------
# Core prediction functions
# ---------------------------------------------------------------------------

def predict_image(model: tf.keras.Model, img_path: Path) -> dict:
    """
    Run inference on a single image.

    Parameters
    ----------
    model    : loaded tf.keras.Model
    img_path : path to the image file

    Returns
    -------
    dict with keys:
        file       : str   — filename only
        result     : str   — "RECYCLABLE" or "ORGANIC"
        confidence : float — confidence percentage (0–100)
        raw_prob   : float — raw sigmoid output (0–1)
    """
    arr = _preprocess_image(img_path)
    raw_prob = float(model.predict(arr, verbose=0)[0][0])

    if raw_prob >= CFG.threshold:
        result     = "RECYCLABLE"
        confidence = raw_prob * 100
    else:
        result     = "ORGANIC"
        confidence = (1.0 - raw_prob) * 100

    return {
        "file"      : img_path.name,
        "result"    : result,
        "confidence": confidence,
        "raw_prob"  : raw_prob,
    }


def predict_from_directory(model: tf.keras.Model, directory: Path) -> tuple:
    """
    Run inference on all images inside a directory.

    Parameters
    ----------
    model     : loaded tf.keras.Model
    directory : path to a folder containing images

    Returns
    -------
    (results, image_paths)
        results     : list[dict]   — one dict per image (from predict_image)
        image_paths : list[Path]   — corresponding file paths
    """
    image_paths = _find_images(directory)
    if not image_paths:
        print(f"⚠️  No images found in: {directory}")
        return [], []

    print(f"\n📸 Found {len(image_paths)} image(s) in: {directory}\n")

    results = []
    for img_path in image_paths:
        result = predict_image(model, img_path)
        results.append(result)

    return results, image_paths


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_predictions(
    results: list,
    image_paths: list,
    save: bool = True,
) -> None:
    """
    Display a grid of images annotated with their predicted class
    and confidence score.

    Each image title is coloured:
      green → RECYCLABLE
      red   → ORGANIC

    Saves to outputs/reports/inference_results.png when save=True.
    """
    if not results:
        return

    num_images = len(results)
    num_cols = 3
    num_rows = (num_images + num_cols - 1) // num_cols  # ceiling division

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    fig.suptitle("Waste Classification — Inference Results", fontsize=14, fontweight="bold")

    # Flatten axes for uniform indexing
    if num_rows == 1 and num_cols == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten().tolist()

    for idx, (res, img_path) in enumerate(zip(results, image_paths)):
        color = "green" if res["result"] == "RECYCLABLE" else "red"
        img_display = keras_image.load_img(str(img_path))
        axes[idx].imshow(img_display)
        axes[idx].set_title(
            f"{res['result']}\n({res['confidence']:.1f}%)",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        axes[idx].axis("off")

    # Hide unused subplot cells
    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save:
        CFG.reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = CFG.reports_dir / "inference_results.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"\n  📊 Inference grid saved → {out_path}")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list) -> None:
    """Print a formatted summary table of all predictions."""
    print("\n" + "=" * 60)
    print("  PREDICTION SUMMARY")
    print("=" * 60)
    print(f"  {'Image Name':<35} {'Prediction':<15} {'Confidence'}")
    print("  " + "-" * 58)

    for r in results:
        print(f"  {r['file']:<35} {r['result']:<15} {r['confidence']:.1f}%")

    print("  " + "-" * 58)

    recyclable_count = sum(1 for r in results if r["result"] == "RECYCLABLE")
    organic_count    = sum(1 for r in results if r["result"] == "ORGANIC")

    print(f"\n  STATISTICS:")
    print(f"    Recyclable : {recyclable_count} image(s)")
    print(f"    Organic    : {organic_count} image(s)")
    print(f"    Total      : {len(results)} image(s)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_inference(model: tf.keras.Model, input_dir: Path) -> list:
    """
    Full inference pipeline in one call:
      1. Find all images in input_dir
      2. Predict each image
      3. Display + save grid visualisation
      4. Print summary table

    Parameters
    ----------
    model     : loaded tf.keras.Model
    input_dir : directory containing images to classify

    Returns
    -------
    results : list[dict]
    """
    results, image_paths = predict_from_directory(model, input_dir)
    if results:
        visualise_predictions(results, image_paths, save=True)
        print_summary(results)
    return results
