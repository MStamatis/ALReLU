#!/usr/bin/env python3
"""Train MNIST with alrelu-keras on Keras + TensorFlow backend."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Force TensorFlow backend for this script before importing keras.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from alrelu_keras import ALReLU, TrainableALReLU
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from alrelu_keras import ALReLU, TrainableALReLU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MNIST using alrelu-keras on Keras/TensorFlow."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument(
        "--variant",
        type=str,
        default="fixed",
        choices=["fixed", "learnable", "trainable"],
        help="ALReLU variant: fixed alpha or learnable alpha.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Initial/fixed alpha value.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def make_activation(variant: str, alpha: float):
    if variant == "fixed":
        return ALReLU(alpha=alpha)
    return TrainableALReLU(alpha_initializer=alpha)


def build_model(variant: str, alpha: float) -> keras.Model:
    return keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, kernel_size=3, padding="same"),
            make_activation(variant, alpha),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(64, kernel_size=3, padding="same"),
            make_activation(variant, alpha),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(128),
            make_activation(variant, alpha),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )


def load_mnist():
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)

    # Hold out last 10k samples as validation.
    x_val, y_val = x_train[-10000:], y_train[-10000:]
    x_train, y_train = x_train[:-10000], y_train[:-10000]
    return x_train, y_train, x_val, y_val


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    variant = "learnable" if args.variant == "trainable" else args.variant

    keras.utils.set_random_seed(args.seed)
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()

    backend_name = keras.backend.backend()
    if backend_name != "tensorflow":
        raise RuntimeError(
            f"This script requires TensorFlow backend, got '{backend_name}'."
        )

    x_train, y_train, x_val, y_val = load_mnist()
    model = build_model(variant=variant, alpha=args.alpha)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    final_val_acc = float(history.history["val_accuracy"][-1])
    print(f"Final val accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
