"""
infer.py — Run inference with a trained HAR model.

Supports all three architectures: 1dcnn, cnn_lstm, wclstm.

Usage:
    # Single sample (1D-CNN only, 19 comma-separated features):
    uv run python -m har_fpga.infer --model 1dcnn --sample "0.28,-0.02,-0.13,..."

    # Evaluate on the UCI HAR test split:
    uv run python -m har_fpga.infer --model 1dcnn --test
    uv run python -m har_fpga.infer --model cnn_lstm --test
    uv run python -m har_fpga.infer --model wclstm --test

The scaler (artifacts/<model>/scaler.json) is applied automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from har_fpga.model import MODEL_TYPES
from har_fpga.preprocess import ZScoreScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _artifact_dir(model_type: str) -> Path:
    return PROJECT_ROOT / "artifacts" / model_type


def _load_class_names() -> list[str]:
    with open(CONFIG_DIR / "training.json", "r") as f:
        return json.load(f)["class_names"]


def _load_training_config() -> dict:
    with open(CONFIG_DIR / "training.json", "r") as f:
        return json.load(f)


def _load_model_and_scaler(
    model_type: str,
    model_path: Path | None = None,
    scaler_path: Path | None = None,
) -> tuple[keras.Model, ZScoreScaler, list[str]]:
    adir = _artifact_dir(model_type)
    model_path = model_path or adir / "har_model.keras"
    scaler_path = scaler_path or adir / "scaler.json"

    if not model_path.exists():
        print(f"[infer] ERROR: Model not found at {model_path}")
        print(
            f"        Train first: uv run python -m har_fpga.train --model {model_type}"
        )
        sys.exit(1)
    if not scaler_path.exists():
        print(f"[infer] ERROR: Scaler not found at {scaler_path}")
        sys.exit(1)

    model = keras.models.load_model(model_path)
    scaler = ZScoreScaler.load(scaler_path)
    class_names = _load_class_names()
    return model, scaler, class_names


def _apply_wavelet_transform(
    X: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
) -> np.ndarray:
    """Apply wavelet decomposition to raw inertial signals."""
    import pywt

    N, T, C = X.shape
    sample_signal = X[0, :, 0]
    coeffs = pywt.wavedec(sample_signal, wavelet, level=level)
    concat_sample = np.concatenate(coeffs)
    T_wt = len(concat_sample)

    X_wt = np.empty((N, T_wt, C), dtype=np.float32)
    for i in range(N):
        for ch in range(C):
            coeffs = pywt.wavedec(X[i, :, ch], wavelet, level=level)
            X_wt[i, :, ch] = np.concatenate(coeffs)
    return X_wt


def _preprocess_for_model(
    X_raw: np.ndarray,
    scaler: ZScoreScaler,
    model_type: str,
    cfg: dict,
) -> np.ndarray:
    """Preprocess raw data for a specific model type.

    Parameters
    ----------
    X_raw : ndarray
        Raw feature vectors (N, 19) for 1dcnn, or raw signals (N, 128, 9).
    scaler : ZScoreScaler
    model_type : str
    cfg : dict
        The model-specific config from training.json.

    Returns
    -------
    X : ndarray ready for model.predict()
    """
    data_mode = cfg.get("data_mode", "features")

    if data_mode == "wavelet":
        wavelet = cfg.get("wavelet", "db4")
        level = cfg.get("wavelet_level", 2)
        X_raw = _apply_wavelet_transform(X_raw, wavelet=wavelet, level=level)

    if X_raw.ndim == 3:
        N, T, C = X_raw.shape
        X_2d = X_raw.reshape(N, T * C)
        X_2d = scaler.transform(X_2d)
        X = X_2d.reshape(N, T, C)
        # For 2D-CNN: add channel dimension (N, T, C) -> (N, T, C, 1)
        if model_type == "2dcnn":
            X = X[..., np.newaxis]
    else:
        X = scaler.transform(X_raw)
        if model_type == "1dcnn":
            X = X[..., np.newaxis]  # (N, 19, 1) for 1D-CNN
        # MLP keeps flat (N, 19) shape

    return X


# -----------------------------------------------------------------------
# Inference functions
# -----------------------------------------------------------------------
def predict_single(
    raw_features: np.ndarray,
    model: keras.Model,
    scaler: ZScoreScaler,
    class_names: list[str],
) -> dict:
    """Predict class for a single raw feature vector (19,).

    Only works for 1dcnn model.
    Returns dict with keys: predicted_class, class_name, probabilities.
    """
    x = raw_features.reshape(1, -1)  # (1, 19)
    x = scaler.transform(x)  # z-score
    x = x[..., np.newaxis]  # (1, 19, 1)
    probs = model.predict(x, verbose=0)[0]  # (5,)
    pred = int(np.argmax(probs))
    return {
        "predicted_class": pred,
        "class_name": class_names[pred],
        "probabilities": {cn: float(p) for cn, p in zip(class_names, probs)},
    }


def predict_batch(
    X: np.ndarray,
    model: keras.Model,
    class_names: list[str],
) -> np.ndarray:
    """Predict classes for a preprocessed batch.

    Returns array of predicted class indices (N,).
    """
    probs = model.predict(X, verbose=0)
    return np.argmax(probs, axis=1)


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained HAR model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="1dcnn",
        choices=list(MODEL_TYPES),
        help="Model architecture (default: 1dcnn)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample",
        type=str,
        help="Comma-separated 19 feature values for single prediction (1dcnn only)",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to text file with samples (one per line)",
    )
    group.add_argument(
        "--test",
        action="store_true",
        help="Run inference on the UCI HAR test split and print accuracy",
    )
    args = parser.parse_args()

    model_type = args.model
    model, scaler, class_names = _load_model_and_scaler(model_type)
    cfg = _load_training_config()
    model_cfg = cfg["models"][model_type]
    data_mode = model_cfg.get("data_mode", "features")

    print(f"[infer] Model: {model_type.upper()}")

    # ---- Single sample (1dcnn only) ----
    if args.sample:
        if model_type != "1dcnn":
            print(f"[infer] ERROR: --sample is only supported for 1dcnn model.")
            print(f"        For {model_type}, use --test to evaluate on test set.")
            sys.exit(1)
        values = [float(v.strip()) for v in args.sample.split(",")]
        if len(values) != 19:
            print(f"[infer] ERROR: Expected 19 values, got {len(values)}")
            sys.exit(1)
        result = predict_single(np.array(values), model, scaler, class_names)
        print(
            f"\nPredicted activity: {result['class_name']} (class {result['predicted_class']})"
        )
        print("Probabilities:")
        for cn, p in result["probabilities"].items():
            bar = "#" * int(p * 40)
            print(f"  {cn:>12s}: {p:.4f}  {bar}")

    # ---- Batch from file (1dcnn only) ----
    elif args.file:
        if model_type != "1dcnn":
            print(f"[infer] ERROR: --file is only supported for 1dcnn model.")
            sys.exit(1)
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"[infer] ERROR: File not found: {filepath}")
            sys.exit(1)
        data = np.loadtxt(filepath, delimiter=None)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != 19:
            print(f"[infer] ERROR: Expected 19 columns, got {data.shape[1]}")
            sys.exit(1)
        X = _preprocess_for_model(data, scaler, model_type, model_cfg)
        preds = predict_batch(X, model, class_names)
        for i, pred in enumerate(preds):
            print(f"  Sample {i:4d}: {class_names[pred]} (class {pred})")
        print(f"\nTotal: {len(preds)} samples")

    # ---- Test split evaluation ----
    elif args.test:
        if data_mode == "features":
            from har_fpga.data import load_har_data

            _, _, X_test_raw, y_test, _, _ = load_har_data(download=False)
        else:
            from har_fpga.data import load_har_raw

            _, _, X_test_raw, y_test, _, _ = load_har_raw(download=False)

        X_test = _preprocess_for_model(X_test_raw, scaler, model_type, model_cfg)
        preds = predict_batch(X_test, model, class_names)
        acc = np.mean(preds == y_test)
        print(
            f"\nTest set accuracy: {acc:.4f} ({np.sum(preds == y_test)}/{len(y_test)})"
        )

        # Per-class breakdown
        print("\nPer-class accuracy:")
        for i, cn in enumerate(class_names):
            mask = y_test == i
            if mask.sum() == 0:
                print(f"  {cn:>12s}: no samples")
            else:
                class_acc = np.mean(preds[mask] == i)
                print(
                    f"  {cn:>12s}: {class_acc:.4f} ({np.sum(preds[mask] == i)}/{mask.sum()})"
                )


if __name__ == "__main__":
    main()
