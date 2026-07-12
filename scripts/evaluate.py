"""
scripts/evaluate.py
-------------------
Entry point for standalone evaluation of a saved model.

Loads an already-trained model and re-runs the full evaluation suite
on the test set without retraining.

Usage
-----
    # Default: loads outputs/models/waste_classifier.keras
    python scripts/evaluate.py

    # Custom model path:
    python scripts/evaluate.py --model path/to/your_model.keras

Colab usage
-----------
    !python scripts/evaluate.py
    !python scripts/evaluate.py --model outputs/models/waste_classifier.keras
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tensorflow as tf

from waste_classifier.config import CFG
from waste_classifier.data import get_test_flow
from waste_classifier.evaluate import run_full_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained waste classification model."
    )
    parser.add_argument(
        "--model",
        type    = str,
        default = str(CFG.model_path),
        help    = f"Path to the saved model (default: {CFG.model_path})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)

    print("=" * 60)
    print("  WASTE SORTING SYSTEM — EVALUATION PIPELINE")
    print("=" * 60)
    print(f"  Model      : {model_path}")
    print(f"  Test set   : {CFG.test_dir}")
    print(f"  Reports    : {CFG.reports_dir}")
    print("=" * 60)

    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Train the model first with:  python scripts/train.py")
        sys.exit(1)

    print("\n[1/2] Loading model...")
    model = tf.keras.models.load_model(str(model_path))

    print("\n[2/2] Loading test split only...")
    # get_test_flow() loads only the test split — no need to load train/val
    # when evaluating a saved model.
    test_data = get_test_flow()

    # History is not available when evaluating a saved model, so we pass
    # None — training curves are skipped automatically.
    run_full_evaluation(model, history=None, test_data=test_data)

    print("\n  Evaluation complete.")


if __name__ == "__main__":
    main()
