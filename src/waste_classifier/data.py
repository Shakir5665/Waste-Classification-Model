"""
data.py
-------
Data pipeline for the three-split dataset (Train / Validation / Test).

Industry-standard practices applied
-------------------------------------
1.  Split validation
    Each directory is checked for existence before any Keras code runs.
    A missing split raises a clear FileNotFoundError with a helpful message.

2.  Class-index assertion
    Keras assigns class indices alphabetically.  After loading each split,
    the class→index mapping is asserted against the config's class_names.
    If a folder is misnamed (e.g. "organic" vs "Organic"), the error is
    caught here — not silently inverted in the evaluation metrics.

3.  Reproducible shuffling
    Every flow_from_directory call receives the same seed (config.yaml → seed: 42).
    Training data is shuffled; validation and test data are NOT shuffled.
    shuffle=False on test is mandatory for the confusion matrix to align.

4.  Augmentation policy
    ONLY the training split receives augmentation.
    Validation and test receive rescaling only.
    This is the correct ML practice — augmenting val/test would give
    unreliable performance estimates.

5.  Dataset statistics
    print_dataset_stats() logs the class distribution for every split
    before training starts.  This is the first line of defence against
    class imbalance bugs.

6.  Split-specific loaders
    Individual functions (load_train_data, load_val_data, load_test_data)
    are provided alongside the combined get_data_flows().
    This lets scripts/evaluate.py and scripts/predict.py load only what
    they need without loading all three splits.

Public API
----------
    get_generators()                          → (train_gen, val_gen, test_gen)
    load_train_data(train_gen)                → DirectoryIterator
    load_val_data(val_gen)                    → DirectoryIterator
    load_test_data(test_gen)                  → DirectoryIterator
    get_data_flows()                          → (train_data, val_data, test_data)
    print_dataset_stats(train, val, test)     → None
    validate_splits()                         → None   (raises on error)
"""

import os
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from waste_classifier.config import CFG


# ---------------------------------------------------------------------------
# 1.  Split validation
# ---------------------------------------------------------------------------

def validate_splits() -> None:
    """
    Verify that all three split directories exist and are non-empty.

    Checks performed
    ----------------
    - Train / Validation / Test directories exist
    - Each directory contains at least one class subdirectory
    - Each class subdirectory contains at least one image file

    Raises
    ------
    FileNotFoundError  if any directory is missing
    ValueError         if a directory is empty or has no images
    """
    splits = {
        "Train"     : CFG.train_dir,
        "Validation": CFG.val_dir,
        "Test"      : CFG.test_dir,
    }

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    for split_name, split_path in splits.items():
        # ── Directory exists ────────────────────────────────────────────
        if not split_path.exists():
            raise FileNotFoundError(
                f"\n\n[DATA ERROR] {split_name} directory not found:\n"
                f"  Expected : {split_path}\n\n"
                f"  Make sure your dataset is structured as:\n"
                f"    dataset/\n"
                f"    ├── train/\n"
                f"    │   ├── organic/\n"
                f"    │   └── recyclable/\n"
                f"    ├── val/\n"
                f"    │   ├── organic/\n"
                f"    │   └── recyclable/\n"
                f"    └── test/\n"
                f"        ├── organic/\n"
                f"        └── recyclable/"
            )

        # ── Has class subdirectories ────────────────────────────────────
        class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        if not class_dirs:
            raise ValueError(
                f"\n\n[DATA ERROR] {split_name} directory is empty (no class subdirectories):\n"
                f"  Path: {split_path}\n"
                f"  Expected subdirectories: {CFG.class_names}"
            )

        # ── Each class dir has images ────────────────────────────────────
        for class_dir in class_dirs:
            images = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ]
            if not images:
                raise ValueError(
                    f"\n\n[DATA ERROR] Class directory has no images:\n"
                    f"  Path     : {class_dir}\n"
                    f"  Expected : .jpg / .jpeg / .png files"
                )


# ---------------------------------------------------------------------------
# 2.  Generator factories
# ---------------------------------------------------------------------------

def get_generators():
    """
    Build and return three ImageDataGenerator instances.

    Augmentation policy (industry standard):
      - Train   : rescale + rotation + zoom + shear + horizontal_flip
      - Val     : rescale only
      - Test    : rescale only
    Augmenting validation/test would give unreliable performance estimates.

    Returns
    -------
    train_gen : ImageDataGenerator  — augmentation enabled
    val_gen   : ImageDataGenerator  — rescale only
    test_gen  : ImageDataGenerator  — rescale only
    """
    train_gen = ImageDataGenerator(
        rescale         = CFG.rescale,
        rotation_range  = CFG.rotation_range,
        zoom_range      = CFG.zoom_range,
        shear_range     = CFG.shear_range,
        horizontal_flip = CFG.horizontal_flip,
    )

    val_gen  = ImageDataGenerator(rescale=CFG.rescale)
    test_gen = ImageDataGenerator(rescale=CFG.rescale)

    return train_gen, val_gen, test_gen


# ---------------------------------------------------------------------------
# 3.  Class-index assertion
# ---------------------------------------------------------------------------

def _assert_class_indices(flow, split_name: str) -> None:
    """
    Verify that the class→index mapping from Keras matches config.yaml.

    Keras assigns class indices alphabetically from the folder names.
    If a folder is misnamed (e.g. "organic" instead of "Organic"), the
    label mapping becomes incorrect — predictions are silently inverted.

    Parameters
    ----------
    flow       : DirectoryIterator returned by flow_from_directory
    split_name : "Train" | "Validation" | "Test"  (for error messages)
    """
    expected = {name: idx for idx, name in enumerate(CFG.class_names)}
    actual   = flow.class_indices

    if actual != expected:
        raise ValueError(
            f"\n\n[DATA ERROR] Class index mismatch in {split_name} split!\n"
            f"  Expected : {expected}\n"
            f"  Got      : {actual}\n\n"
            f"  Cause    : Keras assigns indices alphabetically from folder names.\n"
            f"  Fix      : Rename your class folders to exactly match config.yaml:\n"
            f"             class_names: {CFG.class_names}"
        )


# ---------------------------------------------------------------------------
# 4.  Split-specific loaders
# ---------------------------------------------------------------------------

def load_train_data(train_gen: ImageDataGenerator):
    """
    Load the training split.

    - shuffle=True  (default) for proper SGD gradient estimation
    - seed applied for reproducibility

    Returns
    -------
    DirectoryIterator
    """
    flow = train_gen.flow_from_directory(
        str(CFG.train_dir),
        target_size = CFG.img_size,
        batch_size  = CFG.batch_size,
        class_mode  = "binary",
        shuffle     = True,         # Shuffle training data every epoch
        seed        = CFG.seed,
    )
    _assert_class_indices(flow, "Train")
    return flow


def load_val_data(val_gen: ImageDataGenerator):
    """
    Load the validation split.

    - shuffle=False  (validation order must be deterministic)
    - seed applied for reproducibility

    Returns
    -------
    DirectoryIterator
    """
    flow = val_gen.flow_from_directory(
        str(CFG.val_dir),
        target_size = CFG.img_size,
        batch_size  = CFG.batch_size,
        class_mode  = "binary",
        shuffle     = False,        # Deterministic order for reliable val_loss
        seed        = CFG.seed,
    )
    _assert_class_indices(flow, "Validation")
    return flow


def load_test_data(test_gen: ImageDataGenerator):
    """
    Load the test split.

    - shuffle=False  MANDATORY — the confusion matrix aligns predictions
                     with true labels by position. Any shuffle breaks this.
    - seed applied for reproducibility

    Returns
    -------
    DirectoryIterator
    """
    flow = test_gen.flow_from_directory(
        str(CFG.test_dir),
        target_size = CFG.img_size,
        batch_size  = CFG.batch_size,
        class_mode  = "binary",
        shuffle     = False,        # MUST be False — confusion matrix alignment
        seed        = CFG.seed,
    )
    _assert_class_indices(flow, "Test")
    return flow


# ---------------------------------------------------------------------------
# 5.  Dataset statistics
# ---------------------------------------------------------------------------

def print_dataset_stats(train_data, val_data, test_data) -> None:
    """
    Print a formatted table showing the class distribution for each split.

    Example output
    --------------
    ============================================================
      DATASET STATISTICS
    ============================================================
      Split        Total    Organic    Recyclable   Imbalance
      ─────────────────────────────────────────────────────────
      Train        4,462    2,513      1,949        1.29:1
      Validation     900      531        369        1.44:1
      Test         1,680      138      1,542       11.17:1
      ─────────────────────────────────────────────────────────
      Grand Total  7,042    3,182      3,860        1.21:1
    ============================================================

    Imbalance ratio = majority_class / minority_class.
    A ratio > 3:1 may require class weighting or oversampling.
    """
    splits = [
        ("Train",      train_data),
        ("Validation", val_data),
        ("Test",       test_data),
    ]

    print("\n" + "=" * 62)
    print("  DATASET STATISTICS")
    print("=" * 62)
    print(f"  {'Split':<14} {'Total':>7}  ", end="")
    for name in CFG.class_names:
        print(f"{name:>12}", end="")
    print(f"  {'Imbalance':>10}")
    print("  " + "-" * 58)

    grand_total = 0
    grand_counts = [0] * len(CFG.class_names)

    for split_name, flow in splits:
        total = flow.n
        grand_total += total

        # count images per class from the flow's labels array
        counts = []
        for idx in range(len(CFG.class_names)):
            count = int((flow.classes == idx).sum())
            counts.append(count)
            grand_counts[idx] += count

        max_c = max(counts)
        min_c = min(counts) if min(counts) > 0 else 1
        ratio = max_c / min_c

        line = f"  {split_name:<14} {total:>7,}  "
        for c in counts:
            line += f"{c:>12,}"
        line += f"  {ratio:>8.2f}:1"
        print(line)

    print("  " + "-" * 58)
    grand_max = max(grand_counts)
    grand_min = min(grand_counts) if min(grand_counts) > 0 else 1
    grand_ratio = grand_max / grand_min
    line = f"  {'Grand Total':<14} {grand_total:>7,}  "
    for c in grand_counts:
        line += f"{c:>12,}"
    line += f"  {grand_ratio:>8.2f}:1"
    print(line)

    print("=" * 62)

    # ── Imbalance warning ───────────────────────────────────────────────────
    for split_name, flow in splits:
        counts = [(flow.classes == i).sum() for i in range(len(CFG.class_names))]
        mx, mn = max(counts), min(counts)
        if mn > 0 and mx / mn > 3.0:
            print(
                f"\n  [WARNING] {split_name} split has a class imbalance ratio of "
                f"{mx/mn:.1f}:1.\n"
                f"  Consider using class_weight in model.fit() or oversampling."
            )


# ---------------------------------------------------------------------------
# 6.  Combined loader (used by scripts/train.py)
# ---------------------------------------------------------------------------

def get_data_flows():
    """
    Validate splits, build generators, load all three splits, and print stats.

    This is the single entry point used by the training script.

    Returns
    -------
    train_data : DirectoryIterator  (shuffled, augmented)
    val_data   : DirectoryIterator  (not shuffled, rescale only)
    test_data  : DirectoryIterator  (not shuffled, rescale only)
    """
    # Step 1: validate before loading anything
    validate_splits()

    # Step 2: build generators (augmentation on train only)
    train_gen, val_gen, test_gen = get_generators()

    # Step 3: load each split
    train_data = load_train_data(train_gen)
    val_data   = load_val_data(val_gen)
    test_data  = load_test_data(test_gen)

    # Step 4: print class distribution table
    print_dataset_stats(train_data, val_data, test_data)

    return train_data, val_data, test_data


def get_test_flow():
    """
    Load only the test split.

    Used by scripts/evaluate.py — no need to load train or val
    when only evaluating a saved model.

    Returns
    -------
    test_data : DirectoryIterator  (not shuffled, rescale only)
    """
    # Validate only the test split
    if not CFG.test_dir.exists():
        raise FileNotFoundError(
            f"[DATA ERROR] Test directory not found: {CFG.test_dir}"
        )

    _, _, test_gen = get_generators()
    test_data = load_test_data(test_gen)

    # Print test split stats only
    print("\n" + "=" * 62)
    print("  TEST SPLIT STATISTICS")
    print("=" * 62)
    total = test_data.n
    for idx, name in enumerate(CFG.class_names):
        count = int((test_data.classes == idx).sum())
        print(f"  {name:<20} {count:>6,} images  ({count/total*100:.1f}%)")
    print(f"  {'Total':<20} {total:>6,} images")
    print("=" * 62)

    return test_data
