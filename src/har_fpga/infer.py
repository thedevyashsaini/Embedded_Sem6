"""
infer.py — Run inference with the trained HAR model.

Usage:
    # Single sample from comma-separated values (19 floats):
    uv run python -m har_fpga.infer --sample "0.28,-0.02,-0.13,..."

    # Batch inference from a text file (one sample per line, 19 space/comma-separated values):
    uv run python -m har_fpga.infer --file path/to/samples.txt

    # Run on the test split and print accuracy:
    uv run python -m har_fpga.infer --test

The scaler (artifacts/scaler.json) is applied automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from har_fpga.preprocess import ZScoreScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_class_names() -> list[str]:
    with open(CONFIG_DIR / "training.json", "r") as f:
        return json.load(f)["class_names"]


def _load_model_and_scaler(
    model_path: Path | None = None,
    scaler_path: Path | None = None,
) -> tuple[keras.Model, ZScoreScaler, list[str]]:
    model_path = model_path or ARTIFACT_DIR / "har_model.keras"
    scaler_path = scaler_path or ARTIFACT_DIR / "scaler.json"

    if not model_path.exists():
        print(f"[infer] ERROR: Model not found at {model_path}")
        print("        Train first: uv run python -m har_fpga.train")
        sys.exit(1)
    if not scaler_path.exists():
        print(f"[infer] ERROR: Scaler not found at {scaler_path}")
        sys.exit(1)

    model = keras.models.load_model(model_path)
    scaler = ZScoreScaler.load(scaler_path)
    class_names = _load_class_names()
    return model, scaler, class_names


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
    raw_features: np.ndarray,
    model: keras.Model,
    scaler: ZScoreScaler,
    class_names: list[str],
) -> np.ndarray:
    """Predict classes for a batch of raw feature vectors (N, 19).

    Returns array of predicted class indices (N,).
    """
    x = scaler.transform(raw_features)
    x = x[..., np.newaxis]
    probs = model.predict(x, verbose=0)
    return np.argmax(probs, axis=1)


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with the trained HAR 1D-CNN model."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample",
        type=str,
        help="Comma-separated 19 feature values for single prediction",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to text file with samples (one per line, 19 values each)",
    )
    group.add_argument(
        "--test",
        action="store_true",
        help="Run inference on the UCI HAR test split and print accuracy",
    )
    args = parser.parse_args()

    model, scaler, class_names = _load_model_and_scaler()

    # ---- Single sample ----
    if args.sample:
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

    # ---- Batch from file ----
    elif args.file:
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
        preds = predict_batch(data, model, scaler, class_names)
        for i, pred in enumerate(preds):
            print(f"  Sample {i:4d}: {class_names[pred]} (class {pred})")
        print(f"\nTotal: {len(preds)} samples")

    # ---- Test split evaluation ----
    elif args.test:
        from har_fpga.data import load_har_data

        _, _, X_test, y_test, _, _ = load_har_data(download=False)
        preds = predict_batch(X_test, model, scaler, class_names)
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
