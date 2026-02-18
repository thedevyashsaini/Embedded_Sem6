"""
train.py — Training pipeline for HAR models (1D-CNN, CNN+LSTM, WCLSTM).

Usage:
    uv run python -m har_fpga.train --model 1dcnn                  # default 1D-CNN
    uv run python -m har_fpga.train --model cnn_lstm               # CNN+LSTM (DCLSTM)
    uv run python -m har_fpga.train --model wclstm                 # Wavelet CNN+LSTM
    uv run python -m har_fpga.train --model cnn_lstm --epochs 50 --batch-size 32
    uv run python -m har_fpga.train --model 1dcnn --no-gpu         # force CPU

Steps executed:
  1. Download & load UCI HAR data (features or raw inertial signals)
  2. Apply wavelet transform if WCLSTM
  3. Fit z-score scaler on training set, transform train+test
  4. Build model
  5. Train with Adam + sparse categorical cross-entropy
  6. Evaluate on test set
  7. Save model, scaler, and model spec

Artifacts are written to artifacts/<model_type>/.
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
from har_fpga.data import load_har_data, load_har_raw
from har_fpga.model import build_model, MODEL_TYPES, extract_model_spec
from har_fpga.preprocess import ZScoreScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _artifact_dir(model_type: str) -> Path:
    """Return the artifact directory for a given model type."""
    return PROJECT_ROOT / "artifacts" / model_type


def _check_gpu() -> None:
    """Print GPU availability info."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[train] GPU detected: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[train] No GPU detected. Training will use CPU.")


def _load_training_config() -> dict:
    with open(CONFIG_DIR / "training.json", "r") as f:
        return json.load(f)


def _apply_wavelet_transform(
    X: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
) -> np.ndarray:
    """Apply wavelet decomposition to raw inertial signals.

    Parameters
    ----------
    X : ndarray of shape (N, 128, 9)
        Raw inertial signals (128 timesteps, 9 channels).
    wavelet : str
        Wavelet name (default: 'db4').
    level : int
        Decomposition level (default: 2).

    Returns
    -------
    X_wt : ndarray of shape (N, T_wt, 9)
        Wavelet-transformed signals. The time dimension T_wt is the
        concatenation of approximation + detail coefficients at each level.
    """
    import pywt

    N, T, C = X.shape
    # Determine output length by running one decomposition
    sample_signal = X[0, :, 0]
    coeffs = pywt.wavedec(sample_signal, wavelet, level=level)
    # Concatenate all coefficient arrays: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    concat_sample = np.concatenate(coeffs)
    T_wt = len(concat_sample)
    print(
        f"[train] Wavelet '{wavelet}' level={level}: {T} timesteps -> {T_wt} coefficients per channel"
    )

    X_wt = np.empty((N, T_wt, C), dtype=np.float32)
    for i in range(N):
        for ch in range(C):
            coeffs = pywt.wavedec(X[i, :, ch], wavelet, level=level)
            X_wt[i, :, ch] = np.concatenate(coeffs)

    return X_wt


def train(
    model_type: str = "1dcnn",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    no_gpu: bool = False,
) -> None:
    """Run the full training pipeline for a given model type."""

    if model_type not in MODEL_TYPES:
        print(f"[train] ERROR: Unknown model type '{model_type}'.")
        print(f"        Choose from: {MODEL_TYPES}")
        sys.exit(1)

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
    model_cfg = cfg["models"][model_type]
    data_mode = model_cfg["data_mode"]

    tf.random.set_seed(seed)
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"  Model type: {model_type.upper()}")
    print(f"  Data mode:  {data_mode}")
    print(f"  Epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"{'=' * 60}\n")

    # ---- Load data ----
    if data_mode == "features":
        X_train, y_train, X_test, y_test, feat_names, class_names = load_har_data()
    else:
        # raw or wavelet — both start from raw inertial signals
        X_train, y_train, X_test, y_test, feat_names, class_names = load_har_raw()

    # ---- Wavelet transform (for WCLSTM) ----
    if data_mode == "wavelet":
        wavelet = model_cfg.get("wavelet", "db4")
        level = model_cfg.get("wavelet_level", 2)
        X_train = _apply_wavelet_transform(X_train, wavelet=wavelet, level=level)
        X_test = _apply_wavelet_transform(X_test, wavelet=wavelet, level=level)

    # ---- Normalize ----
    scaler = ZScoreScaler()
    original_shape = X_train.shape
    if X_train.ndim == 3:
        # For raw/wavelet: reshape to 2D for scaler, then back
        N_tr, T, C = X_train.shape
        N_te = X_test.shape[0]
        X_train_2d = X_train.reshape(N_tr, T * C)
        X_test_2d = X_test.reshape(N_te, T * C)
        X_train_2d = scaler.fit_transform(X_train_2d)
        X_test_2d = scaler.transform(X_test_2d)
        X_train = X_train_2d.reshape(N_tr, T, C)
        X_test = X_test_2d.reshape(N_te, T, C)
    else:
        # 1D-CNN: 2D input (N, 19)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # ---- Save scaler ----
    artifact_dir = _artifact_dir(model_type)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(artifact_dir / "scaler.json")

    # ---- Reshape for Conv1D if 1D-CNN: (N, 19) -> (N, 19, 1) ----
    if data_mode == "features":
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

    # ---- Build model config (filter out non-model keys) ----
    build_kwargs = {
        k: v
        for k, v in model_cfg.items()
        if k not in ("_comment", "data_mode", "wavelet", "wavelet_level")
    }

    # For WCLSTM, determine actual input_timesteps from data
    if model_type == "wclstm":
        build_kwargs["input_timesteps"] = X_train.shape[1]

    # ---- Build & compile model ----
    model = build_model(model_type=model_type, **build_kwargs)
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
    print(f"  Model:              {model_type.upper()}")
    print(f"  Training accuracy : {train_acc:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")
    print(f"  Test loss         : {test_loss:.4f}")
    print(f"  Total parameters  : {model.count_params()}")
    print(f"{'=' * 50}")

    # ---- Save model ----
    model_path = artifact_dir / "har_model.keras"
    model.save(model_path)
    print(f"[train] Model saved to {model_path}")

    # ---- Save model spec JSON ----
    spec = extract_model_spec(model)
    spec["model_type"] = model_type
    spec["training_info"] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "final_train_accuracy": float(train_acc),
        "final_test_accuracy": float(test_acc),
        "final_test_loss": float(test_loss),
        "training_time_seconds": round(elapsed, 2),
        "data_mode": data_mode,
        "feature_names": feat_names,
        "class_names": class_names,
        "num_classes": len(class_names),
    }
    if data_mode == "wavelet":
        spec["training_info"]["wavelet"] = model_cfg.get("wavelet", "db4")
        spec["training_info"]["wavelet_level"] = model_cfg.get("wavelet_level", 2)
        spec["training_info"]["wavelet_output_timesteps"] = int(X_train.shape[1])

    spec_path = artifact_dir / "model_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[train] Model spec saved to {spec_path}")

    # ---- Save training history ----
    hist_path = artifact_dir / "training_history.json"
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_data, f, indent=2)
    print(f"[train] Training history saved to {hist_path}")


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a HAR model for FPGA deployment."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="1dcnn",
        choices=list(MODEL_TYPES),
        help="Model architecture to train (default: 1dcnn)",
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
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        no_gpu=args.no_gpu,
    )


if __name__ == "__main__":
    main()
