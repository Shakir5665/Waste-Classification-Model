# Waste Classification Model — Complete Step-by-Step Guide

> This guide takes you from zero to a fully trained and tested model using Google Colab.  
> Follow every section in order on your first run.

---

## Table of Contents

1. [Before You Begin — What You Need](#1-before-you-begin--what-you-need)
2. [Repository Setup](#2-repository-setup)
3. [Opening Google Colab](#3-opening-google-colab)
4. [Enabling the GPU Runtime](#4-enabling-the-gpu-runtime)
5. [Step 1 — Clone the Repository into Colab](#5-step-1--clone-the-repository-into-colab)
6. [Step 2 — Install Dependencies](#6-step-2--install-dependencies)
7. [Step 3 — Upload the Dataset](#7-step-3--upload-the-dataset)
8. [Step 4 — Train the Model](#8-step-4--train-the-model)
9. [Step 5 — Understanding the Training Output](#9-step-5--understanding-the-training-output)
10. [Step 6 — Test with Real-World Images](#10-step-6--test-with-real-world-images)
11. [Step 7 — Download Your Results](#11-step-7--download-your-results)
12. [Re-Evaluating a Saved Model (Without Retraining)](#12-re-evaluating-a-saved-model-without-retraining)
13. [Tweaking the Configuration](#13-tweaking-the-configuration)
14. [Troubleshooting](#14-troubleshooting)
15. [Quick Reference — All Commands](#15-quick-reference--all-commands)

---

## 1. Before You Begin — What You Need

| Requirement           | Details                                                                     |
| --------------------- | --------------------------------------------------------------------------- |
| **Google Account**    | Required to use Google Colab                                                |
| **GitHub Account**    | Required to host the repository                                             |
| **Dataset ZIP**       | `dataset_Implementation-2.zip` — contains Train / Validation / Test folders |
| **Real-world images** | Any `.jpg`, `.jpeg`, or `.png` waste images you want to classify            |
| **Browser**           | Chrome or Firefox recommended for Colab                                     |

> **No local Python or GPU installation required.**  
> Everything runs inside Google Colab on a free cloud GPU.

---

## 2. Repository Setup

Before using Colab you must have the repository on GitHub.

### If you already have it on GitHub

Skip to [Section 3](#3-opening-google-colab). Your repository URL should look like:

```
https://github.com/Shakir5665/Waste Classification Model.git
```

### If you need to push it to GitHub for the first time

Open a terminal on your local machine, navigate to the project folder, and run:

```bash
git init
git add .
git commit -m "Initial commit — modular project structure"
git branch -M main
git remote add origin https://github.com/Shakir5665/Waste Classification Model.git
git push -u origin main
```

> After pushing, every change you make locally can be synced to Colab by running `git pull` inside the notebook.

---

## 3. Opening Google Colab

**Option A — Open the launcher notebook directly from GitHub**

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open notebook**
3. Select the **GitHub** tab
4. Paste your repository URL and press Enter
5. Select `notebooks/colab_runner.ipynb`

**Option B — Open Colab and upload the notebook manually**

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `notebooks/colab_runner.ipynb` from your local machine

---

## 4. Enabling the GPU Runtime

> This step is critical. Without a GPU, training will be very slow (30–60× slower).

1. In Colab, click **Runtime** in the top menu bar
2. Click **Change runtime type**
3. Under **Hardware accelerator**, select **T4 GPU**
4. Click **Save**

You will see the runtime restart. You should see "T4" or similar in the top-right corner of the Colab interface once connected.

---

## 5. Step 1 — Clone the Repository into Colab

Run the **first code cell** in `colab_runner.ipynb`.

```python
import os

REPO_URL  = "https://github.com/Shakir5665/Waste Classification Model.git"
REPO_NAME = "Waste Classification Model"
REPO_DIR  = f"/content/{REPO_NAME}"

if os.path.exists(REPO_DIR):
    print("Repository already cloned. Pulling latest changes...")
    %cd {REPO_DIR}
    !git pull
else:
    print("Cloning repository...")
    !git clone {REPO_URL} {REPO_DIR}
    %cd {REPO_DIR}
```

**Expected output:**

```
Cloning into 'Waste Classification Model'...
remote: Enumerating objects: ...
✅ Working directory: /content/Waste Classification Model
```

> If you run this cell a second time in the same session, it automatically runs `git pull` instead of cloning again.

---

## 6. Step 2 — Install Dependencies

Run the **second code cell**.

```python
!pip install -r requirements.txt -q
print("✅ Dependencies installed.")
```

This installs:

| Package        | Version | Purpose                            |
| -------------- | ------- | ---------------------------------- |
| `tensorflow`   | ≥ 2.12  | Model training and inference       |
| `numpy`        | ≥ 1.23  | Array operations                   |
| `matplotlib`   | ≥ 3.7   | Training curves and visualisations |
| `seaborn`      | ≥ 0.12  | Confusion matrix heatmap           |
| `scikit-learn` | ≥ 1.2   | Classification report and metrics  |
| `pyyaml`       | ≥ 6.0   | Reading `config.yaml`              |
| `Pillow`       | ≥ 9.4   | Image loading and preprocessing    |

**Expected output:**

```
✅ Dependencies installed.
```

> Colab already has most of these packages. The `-q` flag suppresses verbose pip output.

---

## 7. Step 3 — Upload the Dataset

Run the **third code cell**. A file picker will appear in the cell output.

```
Please upload your dataset ZIP file
```

1. Click the **Choose Files** button that appears
2. Navigate to your dataset `.zip` file on your computer
3. Select it and wait for the upload to complete
4. The cell automatically extracts it and places it at the correct path

**Expected output:**

```
Extracting dataset.zip...
✅ Dataset placed at: /content/Waste Classification Model/dataset
```

**Required dataset folder structure** (case-sensitive):

```
dataset/
├── train/        ~21,857 images
│   ├── organic/
│   └── recyclable/
├── val/           ~2,471 images
│   ├── organic/
│   └── recyclable/
└── test/          ~2,470 images
    ├── organic/
    └── recyclable/
```

> **Important:** Folder names must be **exactly** `train/`, `val/`, `test/` and class folders must be **exactly** `organic/`, `recyclable/` (all lowercase). The training script validates this before loading any data.

---

## 8. Step 4 — Train the Model

Run the **fourth code cell**.

```python
!python scripts/train.py
```

This single command runs the **complete pipeline**:

```
[1/4] Loading dataset
      ↓
[2/4] Building the CNN model
      ↓
[3/4] Training (up to 20 epochs with EarlyStopping)
      ↓
[4/4] Evaluating on the unseen test set
      ↓
      Saving model    → outputs/models/waste_classifier.keras
      Saving curves   → outputs/reports/training_curves.png
      Saving CM       → outputs/reports/confusion_matrix.png
      Saving report   → outputs/reports/classification_report.txt
```

**Training typically takes 5–15 minutes on a T4 GPU.**

---

## 9. Step 5 — Understanding the Training Output

### Dataset statistics table

The pipeline prints this before training starts:

```
==============================================================
  DATASET STATISTICS
==============================================================
  Split          Total       organic   recyclable   Imbalance
  ------------------------------------------------------------
  Train         21,857      11,104       10,753       1.03:1
  Validation     2,471       1,388        1,083       1.28:1
  Test           2,470       1,388        1,082       1.28:1
  ------------------------------------------------------------
  Grand Total   26,798      13,880       12,918       1.07:1
==============================================================
```

### Dataset loading (Keras confirmation lines)

```
Found 21857 images belonging to 2 classes.   ← Training set
Found 2471 images belonging to 2 classes.    ← Validation set
Found 2470 images belonging to 2 classes.    ← Test set (never seen during training)
```

### Per-epoch progress

```
Epoch 1/20
683/683 ━━━━━━━━━━━━━━━━━━━━ 34s — accuracy: 0.86 — loss: 0.35 — val_accuracy: 0.88 — val_loss: 0.25
Epoch 2/20
683/683 ━━━━━━━━━━━━━━━━━━━━ 25s — accuracy: 0.87 — loss: 0.30 — val_accuracy: 0.89 — val_loss: 0.24
...
```

Each row shows:

- **accuracy** — how well the model performs on the training images this epoch
- **loss** — training loss (lower is better)
- **val_accuracy** — how well it performs on the validation set (the important one)
- **val_loss** — validation loss (EarlyStopping watches this)

### EarlyStopping trigger

```
Epoch 14/20
...
Restoring model weights from the end of the best epoch: 11.
```

This means training stopped early because `val_loss` did not improve for 3 consecutive epochs. The best weights (from epoch 11) are automatically restored.

### Test set evaluation

```
============================================================
  TEST SET EVALUATION
============================================================
53/53 ━━━━━━━━━━━━━━━━━━━━ 3s

  Test Loss     : 0.1412
  Test Accuracy : 95.30%
```

### Classification report

```
                precision    recall  f1-score   support

       organic       0.xx      0.xx      0.xx      1388
    recyclable       0.xx      0.xx      0.xx      1082

      accuracy                           0.95      1680
     macro avg       0.83      0.89      0.86      1680
  weighted avg       0.96      0.95      0.95      1680
```

**What these numbers mean:**

| Metric        | Definition                                           | Our result      |
| ------------- | ---------------------------------------------------- | --------------- |
| **Precision** | Of all images predicted as X, how many truly are X   | Recyclable: 98% |
| **Recall**    | Of all true X images, how many did we correctly find | Recyclable: 97% |
| **F1-score**  | Harmonic mean of precision and recall                | Recyclable: 97% |
| **Accuracy**  | Overall correct predictions                          | **95.30%**      |

---

## 10. Step 6 — Test with Real-World Images

Run the **fifth code cell**.

```python
!python scripts/predict.py --input Realworld_data/
```

This runs inference on every image inside the `Realworld_data/` folder (24 images are already included in the repository).

### What it does

1. Loads each image from `Realworld_data/`
2. Resizes to 128×128 and normalises pixel values
3. Passes through the trained model
4. Applies the classification threshold (0.5):
   - **Probability ≥ 0.5** → `RECYCLABLE` _(label shown in green)_
   - **Probability < 0.5** → `ORGANIC` _(label shown in red)_
5. Displays a grid of all images with labels and confidence %
6. Prints a summary table

### Using your own images

To test on your own waste photos:

1. Create a folder with your images (`.jpg`, `.jpeg`, or `.png`)
2. Upload that folder to Colab (drag and drop into the file browser on the left)
3. Run:

```python
!python scripts/predict.py --input /content/your_folder_name/
```

### Example summary table output

```
============================================================
  PREDICTION SUMMARY
============================================================
  Image Name                          Prediction      Confidence
  ----------------------------------------------------------
  bottles.jpg                         RECYCLABLE      94.7%
  banana.jpg                          ORGANIC         88.3%
  metal_waste_1.jpg                   RECYCLABLE      97.1%
  bread1.jpg                          ORGANIC         91.5%
  polythene_waste_1.jpg               RECYCLABLE      89.6%
  ...
  ----------------------------------------------------------

  STATISTICS:
    Recyclable : 14 image(s)
    Organic    : 10 image(s)
    Total      : 24 image(s)
============================================================
```

---

## 11. Step 7 — Download Your Results

Run the **sixth code cell** to download all outputs to your computer.

```python
from google.colab import files

# Downloads the trained model
files.download("outputs/models/waste_classifier.keras")

# Downloads the evaluation reports
files.download("outputs/reports/training_curves.png")
files.download("outputs/reports/confusion_matrix.png")
files.download("outputs/reports/classification_report.txt")
```

### What each output file contains

| File                        | Contents                                                                                |
| --------------------------- | --------------------------------------------------------------------------------------- |
| `waste_classifier.keras`    | The trained model weights — use this for future inference                               |
| `training_curves.png`       | Two plots side by side: accuracy curve and loss curve (train vs. validation, per epoch) |
| `confusion_matrix.png`      | Heatmap showing correct/incorrect predictions per class                                 |
| `classification_report.txt` | Full precision, recall, F1-score table saved as plain text                              |

> **Keep `waste_classifier.keras` safe.** It is the result of your entire training run. If you lose it, you have to retrain the model.

---

## 12. Re-Evaluating a Saved Model (Without Retraining)

If you have already trained the model and just want to re-run evaluation:

```python
!python scripts/evaluate.py
```

Or with a specific model file:

```python
!python scripts/evaluate.py --model outputs/models/waste_classifier.keras
```

This skips training entirely and runs:

- Test-set accuracy and loss
- Confusion matrix
- Classification report

---

## 13. Tweaking the Configuration

All settings are stored in [`config.yaml`](../config.yaml) at the root of the repository.  
**You never need to edit source code to change training parameters.**

### Common adjustments

**Train for more epochs:**

```yaml
training:
  epochs: 30 # was 20
```

**Increase patience (allow more epochs before early stopping):**

```yaml
callbacks:
  early_stopping:
    patience: 5 # was 3
```

**Use a larger batch size (if your GPU has enough memory):**

```yaml
data:
  batch_size: 64 # was 32
```

**Change the classification threshold:**

```yaml
inference:
  threshold: 0.6 # was 0.5 — now more conservative about "Recyclable"
```

> After editing `config.yaml` locally, push the change to GitHub and re-run **Step 1** (git pull) in Colab before training again.

---

## 14. Troubleshooting

### "No module named tensorflow"

The GPU runtime was not enabled or the dependencies cell was not run.  
→ Check Section 4, then re-run the install cell.

### "Dataset directory not found"

The dataset upload step was skipped or the ZIP was extracted to the wrong path.  
→ Re-run Step 3. Make sure the file picker shows `dataset_Implementation-2.zip`.

### "Model not found — train the model first"

You ran `scripts/evaluate.py` or `scripts/predict.py` before completing training.  
→ Complete Step 4 first, or upload a previously trained `waste_classifier.keras` to `outputs/models/`.

### "No images found in Realworld_data/"

The `Realworld_data/` folder is empty or the path is wrong.  
→ Confirm images exist: in a Colab cell run `!ls Realworld_data/`  
→ Or pass a full path: `!python scripts/predict.py --input /content/Waste Classification Model/Realworld_data/`

### Colab session disconnected mid-training

Colab sessions disconnect after ~90 minutes of inactivity. If this happens:

1. Re-run all cells from Step 1 (git pull will restore the code)
2. Re-upload the dataset (Step 3)
3. Re-run training (Step 4)

### Training accuracy is much lower than 95%

- Verify the dataset was extracted correctly (Train/Validation/Test subfolders exist)
- Verify the GPU runtime is active (check top-right of Colab — should show T4)
- The model reaches ~95% by epoch 11–14. If EarlyStopping fires at epoch 3–4, check that `val_loss` is decreasing in early epochs.

---

## 15. Quick Reference — All Commands

### Full workflow (run in order)

```bash
# 1. Clone repo (run once per session)
git clone https://github.com/Shakir5665/Waste Classification Model.git
cd Waste Classification Model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (includes evaluation automatically)
python scripts/train.py

# 4. Test with real-world images
python scripts/predict.py --input Realworld_data/
```

### Individual commands

```bash
# Evaluate a saved model without retraining
python scripts/evaluate.py

# Evaluate a model at a custom path
python scripts/evaluate.py --model outputs/models/waste_classifier.keras

# Run inference on a custom image directory
python scripts/predict.py --input /path/to/your/images/

# Run inference with a specific model
python scripts/predict.py --input Realworld_data/ --model outputs/models/waste_classifier.keras
```

### Output locations

| Output                            | Path                                        |
| --------------------------------- | ------------------------------------------- |
| Trained model                     | `outputs/models/waste_classifier.keras`     |
| Training curves (accuracy + loss) | `outputs/reports/training_curves.png`       |
| Confusion matrix                  | `outputs/reports/confusion_matrix.png`      |
| Classification report             | `outputs/reports/classification_report.txt` |
| Inference grid                    | `outputs/reports/inference_results.png`     |

---

_Part of the Waste Sorting System — ICT 3212 Intelligent Systems, Team TECH DREAMERS._

