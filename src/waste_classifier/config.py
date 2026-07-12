"""
config.py
---------
Loads config.yaml and exposes a single `CFG` object used throughout
every module.  Import pattern:

    from waste_classifier.config import CFG
    print(CFG.img_size)          # (128, 128)
    print(CFG.dataset_dir)       # "dataset"
"""

import os
from pathlib import Path
from types import SimpleNamespace

import yaml

# ---------------------------------------------------------------------------
# Locate config.yaml relative to the repository root.
# Works regardless of which directory the script is run from.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]   # src/waste_classifier → src → repo root
_CONFIG_FILE = _REPO_ROOT / "config.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _build_cfg(raw: dict) -> SimpleNamespace:
    """
    Flatten the nested YAML into a single SimpleNamespace with
    convenience attributes so call-sites stay readable.
    """
    d = raw["data"]
    aug = raw["augmentation"]
    tr = raw["training"]
    cb = raw["callbacks"]["early_stopping"]
    m = raw["model"]
    out = raw["outputs"]
    inf = raw["inference"]
    ck = raw["callbacks"]["checkpoint"]
    gd = raw["gdrive"]

    cfg = SimpleNamespace(
        # ── data ────────────────────────────────────────────────────────
        dataset_dir      = _REPO_ROOT / d["dataset_dir"],
        train_dir        = _REPO_ROOT / d["dataset_dir"] / d["train_dir"],
        val_dir          = _REPO_ROOT / d["dataset_dir"] / d["val_dir"],
        test_dir         = _REPO_ROOT / d["dataset_dir"] / d["test_dir"],
        img_size         = tuple(d["img_size"]),
        batch_size       = int(d["batch_size"]),
        class_names      = d["class_names"],           # ["Organic", "Recyclable"]
        seed             = int(d["seed"]),             # reproducibility

        # ── augmentation ────────────────────────────────────────────────
        rescale          = float(aug["rescale"]),
        rotation_range   = int(aug["rotation_range"]),
        zoom_range       = float(aug["zoom_range"]),
        shear_range      = float(aug["shear_range"]),
        horizontal_flip  = bool(aug["horizontal_flip"]),

        # ── training ────────────────────────────────────────────────────
        epochs           = int(tr["epochs"]),
        optimizer        = tr["optimizer"],
        loss             = tr["loss"],
        metrics          = tr["metrics"],

        # ── early stopping ──────────────────────────────────────────────
        es_monitor       = cb["monitor"],
        es_patience      = int(cb["patience"]),
        es_restore       = bool(cb["restore_best_weights"]),

        # ── checkpoint ──────────────────────────────────────────────────
        ckpt_monitor          = ck["monitor"],
        ckpt_best_filename    = ck["save_best_only_filename"],
        ckpt_epoch_template   = ck["save_every_epoch_filename"],

        # ── model architecture ──────────────────────────────────────────
        filters          = list(m["filters"]),
        dense_units      = int(m["dense_units"]),
        dropout_rate     = float(m["dropout_rate"]),
        output_activation= m["output_activation"],

        # ── outputs ─────────────────────────────────────────────────────
        outputs_dir      = _REPO_ROOT / out["dir"],
        models_dir       = _REPO_ROOT / out["models_dir"],
        reports_dir      = _REPO_ROOT / out["reports_dir"],
        checkpoints_dir  = _REPO_ROOT / out["checkpoints_dir"],
        model_path       = _REPO_ROOT / out["models_dir"] / out["model_filename"],
        best_model_path  = _REPO_ROOT / out["models_dir"] / out["best_model_filename"],
        curves_path      = _REPO_ROOT / out["reports_dir"] / out["curves_filename"],
        cm_path          = _REPO_ROOT / out["reports_dir"] / out["confusion_matrix_filename"],
        cm_norm_path     = _REPO_ROOT / out["reports_dir"] / out["confusion_matrix_norm_filename"],
        report_path      = _REPO_ROOT / out["reports_dir"] / out["classification_report_filename"],
        roc_path         = _REPO_ROOT / out["reports_dir"] / out["roc_curve_filename"],
        pr_path          = _REPO_ROOT / out["reports_dir"] / out["pr_curve_filename"],
        metrics_bar_path = _REPO_ROOT / out["reports_dir"] / out["metrics_bar_filename"],
        summary_path     = _REPO_ROOT / out["reports_dir"] / out["training_summary_filename"],

        # ── inference ───────────────────────────────────────────────────
        threshold        = float(inf["threshold"]),
        realworld_dir    = _REPO_ROOT / inf["realworld_data_dir"],

        # ── google drive sync ────────────────────────────────────────────
        drive_sync              = bool(gd["drive_sync"]),
        gdrive_checkpoints_dir  = Path(gd["gdrive_checkpoints_dir"]),
        gdrive_best_model_dir   = Path(gd["gdrive_best_model_dir"]),
    )
    return cfg


# Public singleton — import this everywhere
CFG: SimpleNamespace = _build_cfg(_load_yaml(_CONFIG_FILE))
