"""
model.py
--------
CNN architecture definition.

Preserves the exact model from Implementation 2:
  Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
  → Flatten → Dense(128) → Dropout(0.5) → Dense(1, sigmoid)

Total parameters: 3,304,769 (12.61 MB)

Public API
----------
    build_model()  → compiled tf.keras.Model
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
)

from waste_classifier.config import CFG


def build_model() -> tf.keras.Model:
    """
    Build and compile the waste classification CNN.

    Architecture (preserved from notebook Implementation 2):
      - 3 convolutional blocks with increasing filter depth
      - Global flatten into a fully connected head
      - Dropout regularisation before the output neuron
      - Sigmoid activation for binary classification

    Returns
    -------
    model : compiled tf.keras.Model
    """
    f = CFG.filters  # [32, 64, 128]

    model = Sequential(
        [
            # ── Input ────────────────────────────────────────────────────
            Input(shape=(*CFG.img_size, 3)),

            # ── Block 1 ──────────────────────────────────────────────────
            Conv2D(f[0], (3, 3), activation="relu"),
            MaxPooling2D(2, 2),

            # ── Block 2 ──────────────────────────────────────────────────
            Conv2D(f[1], (3, 3), activation="relu"),
            MaxPooling2D(2, 2),

            # ── Block 3 ──────────────────────────────────────────────────
            Conv2D(f[2], (3, 3), activation="relu"),
            MaxPooling2D(2, 2),

            # ── Head ─────────────────────────────────────────────────────
            Flatten(),
            Dense(CFG.dense_units, activation="relu"),
            Dropout(CFG.dropout_rate),
            Dense(1, activation=CFG.output_activation),
        ],
        name="waste_classifier_v2",
    )

    model.compile(
        optimizer = CFG.optimizer,
        loss      = CFG.loss,
        metrics   = CFG.metrics,
    )

    return model
