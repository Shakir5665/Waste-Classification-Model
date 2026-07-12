"""
scripts/predict.py
------------------
Entry point for running inference on real-world images.

Accepts a directory of images (.jpg, .jpeg, .png) and outputs:
  - A grid visualisation saved to outputs/reports/inference_results.png
  - A printed summary table

Usage
-----
    # Default: uses Realworld_data/ from config.yaml
    python scripts/predict.py

    # Custom input directory:
    python scripts/predict.py --input path/to/your/images/

    # Custom model path:
    python scripts/predict.py --input Realworld_data/ --model outputs/models/waste_classifier.keras

Colab usage
-----------
    !python scripts/predict.py --input Realworld_data/
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tensorflow as tf

from waste_classifier.config import CFG
from waste_classifier.predict import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a directory of waste images."
    )
    parser.add_argument(
        "--input",
        type    = str,
        default = str(CFG.realworld_dir),
        help    = f"Directory containing images to classify (default: {CFG.realworld_dir})",
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
    input_dir  = Path(args.input)
    model_path = Path(args.model)

    print("=" * 60)
    print("  WASTE SORTING SYSTEM — INFERENCE PIPELINE")
    print("=" * 60)
    print(f"  Input dir  : {input_dir}")
    print(f"  Model      : {model_path}")
    print(f"  Threshold  : {CFG.threshold} (≥ threshold → Recyclable)")
    print("=" * 60)

    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Train the model first with:  python scripts/train.py")
        sys.exit(1)

    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir}")
        sys.exit(1)

    print("\n[1/2] Loading model...")
    model = tf.keras.models.load_model(str(model_path))

    print("\n[2/2] Running inference...")
    run_inference(model, input_dir)

    print("\n✅ Inference complete.")


if __name__ == "__main__":
    main()
