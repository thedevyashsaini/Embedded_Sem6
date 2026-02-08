"""
train.py — Training pipeline for the HAR 1D-CNN.

Usage:
    uv run python -m har_fpga.train                          # defaults
    uv run python -m har_fpga.train --epochs 50 --batch-size 32
    uv run python -m har_fpga.train --no-gpu                 # force CPU

Steps executed:
  1. Download & load UCI HAR data (data.py)
  2. Select 19 features, remap labels (data.py)
  3. Fit z-score scaler on training set, transform train+test (preprocess.py)
  4. Build model (model.py)
  5. Train with Adam + sparse categorical cross-entropy
  6. Evaluate on test set
  7. Save model, scaler, and model spec

All artifacts are written to artifacts/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# ---- local imports ----
from har_fpga.data import load_har_data
from har_fpga.model import build_model, extract_model_spec
from har_fpga.preprocess import ZScoreScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "configs"


def _check_gpu() -> None:
    """Print GPU availability info."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[train] GPU detected: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[train] WARNING: No GPU detected. Training will use CPU.")
        print("        If you have an NVIDIA GPU, ensure CUDA toolkit and")
        print("        cuDNN are installed and visible to TensorFlow.")


def _load_training_config() -> dict:
    with open(CONFIG_DIR / "training.json", "r") as f:
        return json.load(f)


def train(
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    no_gpu: bool = False,
) -> None:
    """Run the full training pipeline."""

    # ---- GPU setup ----
    if no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("[train] GPU disabled by --no-gpu flag.")
    _check_gpu()

    # ---- Load config (use CLI overrides if given) ----
    cfg = _load_training_config()
    epochs = epochs or cfg["epochs"]
    batch_size = batch_size or cfg["batch_size"]
    lr = learning_rate or cfg["learning_rate"]
    seed = cfg["random_seed"]

    tf.random.set_seed(seed)
    np.random.seed(seed)

    print(f"[train] Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    # ---- Load data ----
    X_train, y_train, X_test, y_test, feat_names, class_names = load_har_data()

    # ---- Normalize ----
    scaler = ZScoreScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler for inference / FPGA deployment
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    scaler.save(ARTIFACT_DIR / "scaler.json")

    # ---- Reshape for Conv1D: (N, 19) -> (N, 19, 1) ----
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(f"[train] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[train] X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # ---- Split train into train/val ----
    from sklearn.model_selection import train_test_split

    val_split = cfg["validation_split"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_split,
        random_state=seed,
        stratify=y_train,
    )
    print(f"[train] After split: train={X_tr.shape[0]}, val={X_val.shape[0]}")

    # ---- Build & compile model ----
    model = build_model(**cfg["model"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=cfg["loss"],
        metrics=["accuracy"],
    )
    model.summary()

    # ---- Train ----
    t0 = time.time()
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n[train] Training completed in {elapsed:.1f}s")

    # ---- Evaluate on test set ----
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    train_loss, train_acc = model.evaluate(X_tr, y_tr, verbose=0)

    print(f"\n{'=' * 50}")
    print(f"  Training accuracy : {train_acc:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")
    print(f"  Test loss         : {test_loss:.4f}")
    print(f"{'=' * 50}")

    # ---- Save model ----
    model_path = ARTIFACT_DIR / "har_model.keras"
    model.save(model_path)
    print(f"[train] Model saved to {model_path}")

    # ---- Save model spec JSON ----
    spec = extract_model_spec(model)
    spec["training_info"] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "final_train_accuracy": float(train_acc),
        "final_test_accuracy": float(test_acc),
        "final_test_loss": float(test_loss),
        "training_time_seconds": round(elapsed, 2),
        "feature_names": feat_names,
        "class_names": class_names,
        "num_features": len(feat_names),
        "num_classes": len(class_names),
    }
    spec_path = ARTIFACT_DIR / "model_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[train] Model spec saved to {spec_path}")

    # ---- Save training history ----
    hist_path = ARTIFACT_DIR / "training_history.json"
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_data, f, indent=2)
    print(f"[train] Training history saved to {hist_path}")


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the HAR 1D-CNN model for FPGA deployment."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        no_gpu=args.no_gpu,
    )


if __name__ == "__main__":
    main()
