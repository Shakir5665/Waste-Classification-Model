# Inference Guide

## Overview

The inference pipeline classifies waste images from a local directory.
It does **not** require a ZIP file or any interactive upload widget.

## Run Inference

```bash
# Default: uses Realworld_data/ directory
python scripts/predict.py

# Custom input directory
python scripts/predict.py --input path/to/your/images/

# Custom model
python scripts/predict.py --input Realworld_data/ --model outputs/models/waste_classifier.keras
```

## Supported Image Formats

`.jpg`, `.jpeg`, `.png`

## How It Works

1. **Image loading** — each image is loaded with `keras.preprocessing.image.load_img()`
2. **Preprocessing** — resized to `128×128`, normalised to `[0, 1]`
3. **Prediction** — sigmoid output from the trained CNN
4. **Thresholding:**
   - `probability >= 0.5` → **RECYCLABLE** (green)
   - `probability < 0.5`  → **ORGANIC** (red)
5. **Visualisation** — grid of images with coloured labels and confidence %
6. **Summary table** — filename, prediction, confidence for each image

## Outputs

| File | Description |
|---|---|
| `outputs/reports/inference_results.png` | Grid visualisation of all classified images |
| Console | Formatted summary table with per-image statistics |

## Configuring the Threshold

The classification threshold defaults to `0.5` and can be changed in `config.yaml`:

```yaml
inference:
  threshold: 0.5   # Increase to be more conservative about "Recyclable" predictions
```

## Adding New Images

Simply drop `.jpg`, `.jpeg`, or `.png` files into `Realworld_data/`
(or any other directory) and pass that path to `scripts/predict.py`.
