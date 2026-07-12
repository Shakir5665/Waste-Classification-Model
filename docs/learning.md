# Learning Reference — Technologies, Libraries & Techniques

> A complete reference of every technology, library, and technique used in this project,
> with a short description of what each one is and why it is used here.

---

## Table of Contents

1. [Programming Language](#1-programming-language)
2. [Core ML Framework — TensorFlow & Keras](#2-core-ml-framework--tensorflow--keras)
3. [Neural Network Layers](#3-neural-network-layers)
4. [Model Architecture — CNN](#4-model-architecture--cnn)
5. [Activation Functions](#5-activation-functions)
6. [Loss Function & Optimizer](#6-loss-function--optimizer)
7. [Regularisation Techniques](#7-regularisation-techniques)
8. [Training Callbacks](#8-training-callbacks)
9. [Data Pipeline](#9-data-pipeline)
10. [Data Augmentation](#10-data-augmentation)
11. [Evaluation Metrics & Graphs](#11-evaluation-metrics--graphs)
12. [Supporting Libraries](#12-supporting-libraries)
13. [Project Engineering Concepts](#13-project-engineering-concepts)
14. [File Formats](#14-file-formats)
15. [Development Environment](#15-development-environment)

---

## 1. Programming Language

### Python 3.8+
Python is the dominant language for machine learning and data science.
It has a rich ecosystem of ML libraries, clear syntax, and first-class support
from TensorFlow, NumPy, and scikit-learn.  
**Used for:** every source file in this project.

---

## 2. Core ML Framework — TensorFlow & Keras

### TensorFlow (`tensorflow >= 2.12`)
An open-source deep learning framework developed by Google.
It handles the mathematical operations behind neural networks,
including automatic differentiation (backpropagation),
GPU acceleration, and model serialisation.  
**Used for:** building, compiling, training, and saving the CNN model.

### Keras (part of TensorFlow)
Keras is the high-level API built into TensorFlow.
It provides human-readable building blocks (layers, models, callbacks)
that sit on top of TensorFlow's lower-level operations.  
**Used for:** `Sequential` model, all layers, `ImageDataGenerator`, callbacks.

### `tf.keras.Model`
The base class that all Keras models inherit from.
After training, the model object holds all learned weights
and can be saved, loaded, and used for inference.

### `model.fit()`
The method that runs the training loop.
It feeds batches of images through the network,
computes the loss, and updates weights via backpropagation.  
**Key arguments used:** `validation_data`, `epochs`, `callbacks`.

### `model.evaluate()`
Runs the model on the test set in inference mode (dropout disabled)
and returns loss and accuracy.  
**Used for:** final test-set evaluation after training.

### `model.predict()`
Generates raw output probabilities from the sigmoid neuron
for every image in the test set.  
**Used for:** computing confusion matrix, ROC curve, PR curve.

### `model.save()` / `load_model()`
Saves the full model (architecture + weights + compilation config)
to disk in Keras native format (`.keras`).
The model can be reloaded and used without any code changes.  
**Used for:** saving `waste_classifier.keras` and `best_model.keras`.

---

## 3. Neural Network Layers

### `Conv2D` — 2D Convolutional Layer
The fundamental building block of image recognition networks.
A small filter (kernel) slides across the input image and learns to detect
visual patterns like edges, textures, and shapes.
Each filter learns a different feature.  
**Used as:** 3 convolutional blocks with filter depths `32 → 64 → 128`.  
**Kernel size:** `(3, 3)` — a 3×3 pixel receptive field.

### `MaxPooling2D` — Max Pooling Layer
Reduces the spatial dimensions (height × width) of the feature maps
by taking the maximum value in each pooling window.
This makes the model less sensitive to the exact position of features
and reduces the number of computations in subsequent layers.  
**Used with:** `pool_size=(2, 2)` after every `Conv2D` block.

### `Flatten`
Reshapes the 3D feature map output `(height, width, filters)` into a
1D vector so it can be passed into the Dense (fully connected) layers.  
**Used:** once, between the last `MaxPooling2D` and the first `Dense` layer.

### `Dense` — Fully Connected Layer
Every neuron in a Dense layer is connected to every neuron in the previous layer.
It learns complex combinations of the features extracted by the convolutional blocks.  
**Used as:** `Dense(128, activation='relu')` in the head, and `Dense(1, activation='sigmoid')` as the output.

### `Dropout`
Randomly sets a fraction of neuron outputs to zero during each training step.
This prevents neurons from co-adapting and forces the network to learn
more robust, distributed representations.  
**Used with:** `rate=0.5` (50% of neurons dropped per step, training only).

### `Input`
Explicitly declares the shape of the model's input tensor.
Using `Input(shape=(128, 128, 3))` is the modern Keras practice
instead of passing `input_shape` to the first layer.

---

## 4. Model Architecture — CNN

### Sequential Model
A linear stack of layers where the output of each layer feeds directly
into the next.  Suitable for this project because the data flow is
straightforward: image → convolutional blocks → head → output.

### Convolutional Neural Network (CNN)
A class of deep learning architecture specifically designed for image data.
CNNs exploit the spatial structure of images using convolutional layers
that share weights across positions, making them far more efficient than
fully connected networks on image inputs.

**Architecture used in this project:**

```
Input  (128 × 128 × 3)
  │
Conv2D(32)  → MaxPool      ← detects low-level features: edges, colours
  │
Conv2D(64)  → MaxPool      ← detects mid-level features: corners, textures
  │
Conv2D(128) → MaxPool      ← detects high-level features: object parts
  │
Flatten
  │
Dense(128, relu)
  │
Dropout(0.5)
  │
Dense(1, sigmoid)          ← binary output: organic vs recyclable
```

**Total parameters:** 3,304,769 (12.61 MB)

### Binary Classification
This project classifies each image into one of exactly two classes.
A single output neuron with a sigmoid activation outputs a probability
between 0 and 1, which is compared to a threshold (0.5) to decide the class.

---

## 5. Activation Functions

### ReLU — Rectified Linear Unit
`f(x) = max(0, x)`  
The most widely used activation function in hidden layers.
It introduces non-linearity while avoiding the vanishing gradient problem
that affects sigmoid/tanh in deep networks.  
**Used in:** all `Conv2D` layers and the `Dense(128)` head layer.

### Sigmoid
`f(x) = 1 / (1 + e^(-x))`  
Squashes any real number into the range (0, 1), making it ideal
for binary classification where the output represents a probability.  
**Used in:** the final `Dense(1)` output layer.  
**Interpretation:** output ≥ 0.5 → recyclable, output < 0.5 → organic.

---

## 6. Loss Function & Optimizer

### Binary Cross-Entropy Loss
The standard loss function for binary classification.
It measures the difference between the predicted probability and the true label
(0 or 1).  Penalises confident wrong predictions heavily.  
**Formula:** `L = -(y * log(p) + (1-y) * log(1-p))`  
**Used as:** `loss='binary_crossentropy'` in `model.compile()`.

### Adam Optimizer
Adaptive Moment Estimation.
Combines the benefits of two other optimisers (RMSProp and Momentum).
Adapts the learning rate for each parameter individually,
making it robust and fast-converging with minimal tuning.
It is the default choice for most deep learning projects.  
**Used as:** `optimizer='adam'` in `model.compile()`.

---

## 7. Regularisation Techniques

### Dropout (rate = 0.5)
During each training step, 50% of the neurons in the dropout layer
are randomly deactivated.  This forces the network to not rely on
any single neuron and to spread learning across all neurons.
At inference time, all neurons are active and outputs are scaled automatically.  
**Effect:** reduces overfitting and improves generalisation to unseen data.  
**Location in model:** between `Dense(128)` and `Dense(1, sigmoid)`.

### Data Augmentation
Artificially increases training set diversity by applying random
transformations to each image before it is fed to the model.
The original images on disk are not changed — transformations are
applied on-the-fly during training.  
**Effect:** the model learns features that are invariant to rotation,
zoom, and orientation, which improves real-world performance.  
**See:** [Section 10](#10-data-augmentation) for full details.

### Early Stopping
Monitors a metric (validation loss) during training and stops automatically
when it stops improving.  Prevents the model from memorising the training data.  
**See:** [Section 8](#8-training-callbacks) for full details.

### Explicit Train / Validation / Test Split
The dataset is divided into three non-overlapping subsets:

| Split | Size | Purpose |
|---|---|---|
| `train/` | 21,857 images | Adjust model weights (learning) |
| `val/` | 2,471 images | Tune hyperparameters, monitor early stopping |
| `test/` | 2,470 images | Final honest evaluation — never seen during training |

Using a separate test set prevents the final accuracy number from being
optimistically biased by decisions made during training.

---

## 8. Training Callbacks

Callbacks are functions that run automatically at specific points during training
(end of each epoch, end of training, etc.).

### EarlyStopping
Monitors `val_loss` after every epoch.
If `val_loss` does not improve for `patience=3` consecutive epochs,
training stops automatically.
`restore_best_weights=True` means the model weights are rolled back
to the epoch where `val_loss` was lowest.  
**Benefit:** prevents overtraining; saves time; gives the best possible model.

### ModelCheckpoint — Best Model
Saves the full model to `outputs/models/best_model.keras`
every time `val_loss` reaches a new minimum.
This snapshot is independent of EarlyStopping — it is always available
even if training is interrupted.  
**File saved:** `best_model.keras`

### ModelCheckpoint — Every Epoch
Saves a full model snapshot after every epoch to
`outputs/models/checkpoints/epoch_NN_valloss_X.XXXX.keras`.
This creates a complete training audit trail and allows you to
load and inspect the model at any intermediate point in training.  
**Files saved:** one `.keras` file per epoch.

---

## 9. Data Pipeline

### `ImageDataGenerator` (Keras)
Generates batches of images from a directory structure on disk.
Handles loading, decoding, resizing, and optional augmentation
without loading the entire dataset into memory at once.  
**Used for:** creating train, val, and test data streams.

### `flow_from_directory()`
Reads images from a folder organised as:
```
split/
├── class_a/
└── class_b/
```
Automatically assigns labels based on subfolder names.
Returns a `DirectoryIterator` that yields `(batch_of_images, batch_of_labels)`.  
**Key arguments used:**
- `target_size=(128, 128)` — resizes every image to 128×128 pixels
- `batch_size=32` — 32 images per training step
- `class_mode='binary'` — returns labels as 0.0 or 1.0
- `shuffle=False` on test — required for confusion matrix correctness
- `seed=42` — reproducible batch ordering

### Pixel Normalisation (`rescale = 1/255`)
Converts pixel values from the range [0, 255] to [0.0, 1.0].
Neural networks train more stably when inputs are small floating-point values.

### Batch Processing
Instead of feeding the entire dataset at once (which would not fit in memory),
images are processed in batches of 32.
Each batch produces one gradient update step.

---

## 10. Data Augmentation

Applied to the **training split only**. Validation and test splits receive
only pixel rescaling — augmenting them would make performance metrics unreliable.

| Transformation | Setting | What it does |
|---|---|---|
| `rescale` | 1/255 | Normalise pixel values to [0, 1] |
| `rotation_range` | 20° | Randomly rotate the image up to ±20 degrees |
| `zoom_range` | 0.2 | Randomly zoom in or out by up to 20% |
| `shear_range` | 0.2 | Apply a shear transformation (skew along an axis) |
| `horizontal_flip` | True | Randomly mirror the image left-to-right |

---

## 11. Evaluation Metrics & Graphs

All graphs are saved to `outputs/reports/`.

### Accuracy
The percentage of images classified correctly.  
`accuracy = correct predictions / total predictions`  
**Limitation:** can be misleading when classes are imbalanced.

### Loss (Binary Cross-Entropy)
The value the optimiser minimises during training.
Lower is better.  Tracked on both training and validation sets per epoch.

### Training Curves
Two side-by-side line plots showing train vs. validation accuracy and loss
per epoch.  A vertical dashed line marks the best epoch.  
**File:** `training_curves.png`  
**Why:** reveals overfitting (train accuracy high, val accuracy low gap) and
underfitting (both curves flat at a low value).

### Confusion Matrix (Counts)
A 2×2 table showing:
- **True Positives (TP):** correctly predicted recyclable
- **True Negatives (TN):** correctly predicted organic
- **False Positives (FP):** organic predicted as recyclable
- **False Negatives (FN):** recyclable predicted as organic

**File:** `confusion_matrix.png`

### Confusion Matrix (Normalised)
The same matrix expressed as row-wise percentages.
Each row sums to 100%, showing the per-class accuracy rate directly.  
**File:** `confusion_matrix_normalised.png`

### Precision
Of all images predicted as class X, what fraction actually are class X?  
`precision = TP / (TP + FP)`  
**High precision** means few false alarms.

### Recall (Sensitivity)
Of all true class X images, what fraction did the model correctly identify?  
`recall = TP / (TP + FN)`  
**High recall** means few missed detections.

### F1-Score
The harmonic mean of precision and recall.
A single number that balances both — useful when classes are imbalanced.  
`F1 = 2 * (precision * recall) / (precision + recall)`

### Classification Report
A text table from scikit-learn showing precision, recall, F1-score,
and support (number of samples) for each class, plus overall averages.  
**File:** `classification_report.txt`

### ROC Curve — Receiver Operating Characteristic
Plots the True Positive Rate (recall) against the False Positive Rate
at every possible classification threshold.  
**AUC (Area Under Curve):** a single number between 0.5 and 1.0.
AUC = 1.0 is a perfect classifier.  AUC = 0.5 is random guessing.  
**File:** `roc_curve.png`  
**Why:** gives a threshold-independent view of model quality.

### Precision-Recall (PR) Curve
Plots precision against recall at every threshold.
More informative than the ROC curve when classes are imbalanced,
because it focuses on the minority class performance.  
**Average Precision (AP):** the area under the PR curve.  
**File:** `precision_recall_curve.png`

### Per-Class Metrics Bar Chart
A grouped bar chart showing Precision, Recall, and F1-score
side-by-side for each class.
Makes it visually obvious which class the model struggles with.  
**File:** `per_class_metrics.png`

### Training Summary JSON
A machine-readable file containing all scalar metrics from the training run:
best epoch, val_loss, val_accuracy, test accuracy, ROC AUC, average precision,
and the full per-epoch history arrays.  
**File:** `training_summary.json`  
**Why:** allows programmatic comparison of multiple training runs.

---

## 12. Supporting Libraries

### NumPy (`numpy >= 1.23`)
The fundamental package for numerical computing in Python.
Provides fast N-dimensional arrays and mathematical operations.  
**Used for:** array manipulation, rounding predictions, computing class counts,
imbalance ratios, and metric calculations.

### Matplotlib (`matplotlib >= 3.7`)
The standard Python plotting library.
Produces static, publication-quality figures.  
**Used for:** training curves, ROC curve, PR curve, per-class metrics bar chart,
and the inference image grid.

### Seaborn (`seaborn >= 0.12`)
A statistical visualisation library built on top of Matplotlib.
Provides higher-level plot types with better default aesthetics.  
**Used for:** confusion matrix heatmap (`sns.heatmap()`).

### scikit-learn (`scikit-learn >= 1.2`)
A comprehensive machine learning library for Python.
Used here only for its evaluation utilities — no model training.  
**Used for:**
- `confusion_matrix()` — 2×2 count matrix
- `classification_report()` — precision/recall/F1 table
- `roc_curve()` + `auc()` — ROC curve data
- `precision_recall_curve()` + `average_precision_score()` — PR curve data
- `precision_score()`, `recall_score()`, `f1_score()` — per-class bar chart

### PyYAML (`pyyaml >= 6.0`)
A YAML parser for Python.  
**Used for:** reading `config.yaml` into a Python dictionary at startup.

### Pillow (`Pillow >= 9.4`)
The standard Python image processing library.  
**Used for:** loading images during inference (`keras_image.load_img()` wraps Pillow internally).

### `pathlib.Path` (Python standard library)
An object-oriented interface for filesystem paths.
Safer and more readable than string-based path manipulation.  
**Used throughout:** all file path construction in `config.py`, `data.py`, `train.py`, `evaluate.py`.

### `json` (Python standard library)
Built-in JSON serialisation.  
**Used for:** writing `training_summary.json`.

### `argparse` (Python standard library)
Built-in command-line argument parser.  
**Used for:** `--model` and `--input` arguments in `scripts/evaluate.py` and `scripts/predict.py`.

---

## 13. Project Engineering Concepts

### Modular Package Architecture (`src/waste_classifier/`)
All application logic is split into separate Python modules,
each with a single responsibility.
This follows the **Single Responsibility Principle (SRP)** — a core
software engineering practice that makes code easier to test,
modify, and understand.

### Configuration Management (`config.yaml` + `config.py`)
All hyperparameters and file paths live in one YAML file.
No hardcoded values exist inside source modules.
This follows the **Twelve-Factor App** principle of separating
configuration from code — the model's behaviour can be changed
without editing Python source files.

### Singleton Configuration Object (`CFG`)
`config.py` parses `config.yaml` once at import time and exposes
a single `CFG` object used by every module.
This ensures all modules read from the same configuration
and prevents inconsistencies.

### CLI Entry Points (`scripts/`)
Each workflow (train, evaluate, predict) has its own standalone
Python script that can be run from the command line.
This is the standard way to expose ML pipelines in production
and makes them runnable from both terminal and Colab.

### Seed / Reproducibility
`seed=42` is passed to every random operation.
This ensures that re-running the training script produces
the same results, which is essential for debugging and
scientific reproducibility.

### Dataset Split Validation
The data pipeline checks that all three splits exist and are
non-empty before any model code runs.
Failing early with a clear message is far better than failing
deep inside Keras with a cryptic error.

### Class-Index Assertion
After Keras loads each split, the code verifies that Keras's
alphabetical class assignment matches the order in `config.yaml`.
This prevents silent label inversion — one of the hardest bugs
to detect in classification pipelines.

### Dataset Statistics Logging
The class count per split is printed to the console before training.
This is the first line of defence against class imbalance bugs
and ensures the team understands the data distribution.

### Augmentation Policy (Train Only)
Augmentation is applied exclusively to the training split.
Applying it to validation or test would produce different results
each evaluation run, making the metrics unreliable.

### Non-Interactive Matplotlib Backend (`Agg`)
`matplotlib.use("Agg")` is set before any plotting code.
This forces Matplotlib to render to files without opening a GUI window,
making the code work identically in Colab, scripts, and headless servers.

### `.gitignore`
Prevents large files (model weights, generated plots, Python cache) from
being committed to Git.
Keeps the repository clean and fast to clone.

### `outputs/` Directory Structure
All generated files (model weights, plots, reports) go into a
dedicated `outputs/` directory that is separated from source code.
This makes it easy to clean, archive, or ignore generated artefacts.

---

## 14. File Formats

### `.keras` (Native Keras Format)
The modern format for saving complete Keras models
(architecture + weights + compilation config).
Replaces the older `.h5` (HDF5) format, which Keras now shows
a deprecation warning for.  
**Used for:** `waste_classifier.keras`, `best_model.keras`, epoch checkpoints.

### `.yaml` / `.yml`
A human-readable data serialisation format.
Supports comments, making it ideal for configuration files.  
**Used for:** `config.yaml` — the single source of truth for all settings.

### `.json`
A lightweight data interchange format.  
**Used for:** `training_summary.json` — machine-readable training results.

### `.png`
Lossless image format used for all saved plots and evaluation graphs.

### `.txt`
Plain text, used for saving the classification report so it can be
opened in any editor or pasted into a report.

---

## 15. Development Environment

### Google Colab
A free cloud-based Jupyter notebook environment provided by Google.
Offers free GPU access (T4) which dramatically speeds up CNN training.  
**Used for:** running the training script on GPU via `notebooks/colab_runner.ipynb`.

### GitHub
A web-based platform for hosting Git repositories.
Used to version-control the source code and provide a URL that
Colab can clone from.

### Git
A distributed version control system.
Tracks changes to every source file over time.
Allows the Colab notebook to always pull the latest code with `git pull`.

### Virtual Environment (`.venv/`)
An isolated Python environment that contains only the packages
this project needs, without affecting the system Python installation.  
**Set up with:** `python -m venv .venv`

---

*Part of the Waste Sorting System — ICT 3212 Intelligent Systems, Team TECH DREAMERS.*
