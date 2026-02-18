"""
quantize.py -- Post-training weight quantization comparison for HAR models.

Supports all three architectures: 1dcnn, cnn_lstm, wclstm.

Quantizes the trained FP32 weights to several reduced-precision formats,
runs inference on the UCI HAR test split with each variant, and produces:

  artifacts/<model_type>/quantization/
      results.json              -- accuracy, inference time, weight sizes per variant
      quantization_results.png  -- comparison bar charts
      fp16/                     -- FP16 weights (.mem + metadata.json)
      int16/                    -- INT16 weights (.mem + metadata.json)
      int8/                     -- INT8  weights (.mem + metadata.json)

No retraining is performed. Weights are quantized post-training using
symmetric min-max scaling for integer formats.

Usage:
    uv run python -m har_fpga.quantize --model 1dcnn
    uv run python -m har_fpga.quantize --model cnn_lstm
    uv run python -m har_fpga.quantize --model wclstm
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from har_fpga.data import load_har_data, load_har_raw
from har_fpga.model import MODEL_TYPES
from har_fpga.preprocess import ZScoreScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

# Number of inference timing runs (after warmup)
TIMING_RUNS = 10
WARMUP_RUNS = 3


def _artifact_dir(model_type: str) -> Path:
    return PROJECT_ROOT / "artifacts" / model_type


def _load_training_config() -> dict:
    with open(CONFIG_DIR / "training.json", "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Wavelet transform (reused from train.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Utility: IEEE-754 hex encoding for .mem files
# ---------------------------------------------------------------------------
def _float32_to_hex(value: float) -> str:
    """IEEE-754 float32 -> 8 hex chars."""
    return format(struct.unpack(">I", struct.pack(">f", value))[0], "08x")


def _float16_to_hex(value: np.float16) -> str:
    """IEEE-754 float16 -> 4 hex chars."""
    return format(struct.unpack(">H", struct.pack(">e", value))[0], "04x")


def _int16_to_hex(value: int) -> str:
    """Signed int16 -> 4 hex chars (two's complement)."""
    return format(value & 0xFFFF, "04x")


def _int8_to_hex(value: int) -> str:
    """Signed int8 -> 2 hex chars (two's complement)."""
    return format(value & 0xFF, "02x")


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------
def quantize_symmetric(arr: np.ndarray, n_bits: int) -> tuple[np.ndarray, float]:
    """Symmetric min-max quantization to signed integer.

    Returns (quantized_int_array, scale) where:
        quantized = round(arr / scale)
        dequantized = quantized * scale
    """
    qmin = -(2 ** (n_bits - 1)) + 1
    qmax = 2 ** (n_bits - 1) - 1

    abs_max = max(abs(arr.min()), abs(arr.max()))
    if abs_max == 0:
        return np.zeros_like(arr, dtype=np.int32), 1.0

    scale = abs_max / qmax
    quantized = np.clip(np.round(arr / scale), qmin, qmax).astype(np.int32)
    return quantized, float(scale)


def dequantize(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize back to float32."""
    return (quantized.astype(np.float32)) * scale


# ---------------------------------------------------------------------------
# .mem export per variant
# ---------------------------------------------------------------------------
def _write_mem_fp16(weights_dict: dict[str, np.ndarray], out_dir: Path) -> dict:
    """Convert all weights to FP16, write .mem, return metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("// HAR Model  FP16 quantized weights")
    lines.append("")
    metadata: dict = {"dtype": "float16", "layers": {}}

    for name, arr in weights_dict.items():
        arr16 = arr.astype(np.float16)
        flat = arr16.flatten()
        shape_str = "x".join(str(d) for d in arr.shape)
        lines.append(f"// {name}  shape=({shape_str})  count={flat.size}")
        for v in flat:
            lines.append(_float16_to_hex(v))
        lines.append("")
        metadata["layers"][name] = {
            "shape": list(arr.shape),
            "count": int(flat.size),
        }

    mem_path = out_dir / "model_weights_fp16.mem"
    with open(mem_path, "w") as f:
        f.write("\n".join(lines))
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _write_mem_int(
    weights_dict: dict[str, np.ndarray],
    n_bits: int,
    out_dir: Path,
) -> dict:
    """Quantize to int<n_bits>, write .mem, return metadata with scales."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hex_fn = _int8_to_hex if n_bits == 8 else _int16_to_hex
    label = f"int{n_bits}"

    lines: list[str] = []
    lines.append(f"// HAR Model  {label.upper()} quantized weights (symmetric)")
    lines.append("")
    metadata: dict = {"dtype": label, "layers": {}}

    for name, arr in weights_dict.items():
        q, scale = quantize_symmetric(arr, n_bits)
        flat = q.flatten()
        shape_str = "x".join(str(d) for d in arr.shape)
        lines.append(
            f"// {name}  shape=({shape_str})  count={flat.size}  scale={scale:.10e}"
        )
        for v in flat:
            lines.append(hex_fn(int(v)))
        lines.append("")
        metadata["layers"][name] = {
            "shape": list(arr.shape),
            "count": int(flat.size),
            "scale": scale,
        }

    mem_path = out_dir / f"model_weights_{label}.mem"
    with open(mem_path, "w") as f:
        f.write("\n".join(lines))
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


# ---------------------------------------------------------------------------
# Inference with quantized weights (weight-substitution approach)
# ---------------------------------------------------------------------------
def _extract_weights(model: keras.Model) -> dict[str, np.ndarray]:
    """Extract all trainable weight arrays keyed by layer_name/weight_name."""
    d: dict[str, np.ndarray] = {}
    for layer in model.layers:
        for w in layer.weights:
            key = f"{layer.name}/{w.name}"
            d[key] = w.numpy().astype(np.float32)
    return d


def _set_weights_from_dict(model: keras.Model, wd: dict[str, np.ndarray]) -> None:
    """Set model weights from a name->array dict."""
    for layer in model.layers:
        layer_weights = []
        for w in layer.weights:
            key = f"{layer.name}/{w.name}"
            layer_weights.append(wd[key].astype(np.float32))
        if layer_weights:
            layer.set_weights(layer_weights)


def _build_quantized_weights(
    original: dict[str, np.ndarray],
    variant: str,
) -> dict[str, np.ndarray]:
    """Return a dict of dequantized (back to fp32) weights for a given variant."""
    result: dict[str, np.ndarray] = {}
    for name, arr in original.items():
        if variant == "fp32":
            result[name] = arr.copy()
        elif variant == "fp16":
            result[name] = arr.astype(np.float16).astype(np.float32)
        elif variant == "int16":
            q, s = quantize_symmetric(arr, 16)
            result[name] = dequantize(q, s)
        elif variant == "int8":
            q, s = quantize_symmetric(arr, 8)
            result[name] = dequantize(q, s)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    return result


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def _timed_inference(
    model: keras.Model,
    X: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run inference, return (predictions, avg_time_seconds)."""
    for _ in range(WARMUP_RUNS):
        model.predict(X, verbose=0)

    times = []
    preds = None
    for _ in range(TIMING_RUNS):
        t0 = time.perf_counter()
        preds = model.predict(X, verbose=0)
        times.append(time.perf_counter() - t0)

    avg_time = float(np.mean(times))
    return preds, avg_time  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Weight size calculation
# ---------------------------------------------------------------------------
def _weight_size_bytes(weights_dict: dict[str, np.ndarray], variant: str) -> int:
    """Calculate total weight storage size in bytes for a variant."""
    total_params = sum(arr.size for arr in weights_dict.values())
    bits_per_param = {"fp32": 32, "fp16": 16, "int16": 16, "int8": 8}
    return total_params * bits_per_param[variant] // 8


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _generate_plots(results: list[dict], model_type: str, out_path: Path) -> None:
    """Generate comparison bar charts."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    variants = [r["variant"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    times_ms = [r["inference_time_s"] * 1000 for r in results]
    sizes_bytes = [r["weight_size_bytes"] for r in results]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    total_params = results[0].get("total_params", "?")
    fig.suptitle(
        f"Post-Training Weight Quantization Comparison\n"
        f"HAR {model_type.upper()} ({total_params} parameters)",
        fontsize=14,
        fontweight="bold",
    )

    # --- Accuracy ---
    ax = axes[0]
    bars = ax.bar(variants, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # --- Inference time ---
    ax = axes[1]
    bars = ax.bar(variants, times_ms, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Avg Inference Time (ms)")
    ax.set_title("Inference Time (full test set)")
    for bar, val in zip(bars, times_ms):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times_ms) * 0.02,
            f"{val:.1f}ms",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # --- Weight size ---
    ax = axes[2]
    bars = ax.bar(variants, sizes_bytes, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Weight Size (bytes)")
    ax.set_title("Weight Storage Size")
    for bar, val in zip(bars, sizes_bytes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sizes_bytes) * 0.02,
            f"{val} B",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[quantize] Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-training weight quantization comparison for HAR models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="1dcnn",
        choices=list(MODEL_TYPES),
        help="Model architecture (default: 1dcnn)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained Keras model (overrides default)",
    )
    args = parser.parse_args()

    model_type = args.model
    adir = _artifact_dir(model_type)
    quant_dir = adir / "quantization"

    model_path = Path(args.model_path) if args.model_path else adir / "har_model.keras"
    if not model_path.exists():
        print(f"[quantize] ERROR: Model not found at {model_path}")
        print(
            f"           Train first: uv run python -m har_fpga.train --model {model_type}"
        )
        raise SystemExit(1)

    scaler_path = adir / "scaler.json"
    if not scaler_path.exists():
        print(f"[quantize] ERROR: Scaler not found at {scaler_path}")
        raise SystemExit(1)

    # ---- Load config ----
    cfg = _load_training_config()
    model_cfg = cfg["models"][model_type]
    data_mode = model_cfg.get("data_mode", "features")
    class_names = cfg["class_names"]

    # ---- Load model, scaler, data ----
    print(f"[quantize] Model type: {model_type.upper()}")
    print(f"[quantize] Loading model from {model_path} ...")
    model = keras.models.load_model(model_path)
    scaler = ZScoreScaler.load(scaler_path)

    print("[quantize] Loading test data ...")
    if data_mode == "features":
        _, _, X_test_raw, y_test, _, _ = load_har_data(download=True)
    else:
        _, _, X_test_raw, y_test, _, _ = load_har_raw(download=True)

    # Wavelet transform for WCLSTM
    if data_mode == "wavelet":
        wavelet = model_cfg.get("wavelet", "db4")
        level = model_cfg.get("wavelet_level", 2)
        X_test_raw = _apply_wavelet_transform(X_test_raw, wavelet=wavelet, level=level)

    # Preprocess: z-score + reshape
    if X_test_raw.ndim == 3:
        N, T, C = X_test_raw.shape
        X_2d = X_test_raw.reshape(N, T * C)
        X_2d = scaler.transform(X_2d)
        X_test = X_2d.reshape(N, T, C)
    else:
        X_test = scaler.transform(X_test_raw)
        X_test = X_test[..., np.newaxis]  # (N, 19, 1)

    print(f"[quantize] Test samples: {X_test.shape[0]}")
    print(f"[quantize] Total model parameters: {model.count_params()}")
    print()

    # ---- Extract original FP32 weights ----
    original_weights = _extract_weights(model)

    # ---- Variants to test ----
    variants = ["fp32", "fp16", "int16", "int8"]
    results: list[dict] = []

    quant_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        print(f"{'=' * 60}")
        print(f"  Model: {model_type.upper()}  |  Variant: {variant.upper()}")
        print(f"{'=' * 60}")

        # Build quantized (then dequantized-to-fp32) weights
        qw = _build_quantized_weights(original_weights, variant)

        # Inject into model
        _set_weights_from_dict(model, qw)

        # Run inference
        preds_probs, avg_time = _timed_inference(model, X_test)
        preds = np.argmax(preds_probs, axis=1)
        accuracy = float(np.mean(preds == y_test))
        weight_size = _weight_size_bytes(original_weights, variant)

        # Per-class accuracy
        per_class: dict[str, float] = {}
        for i, cn in enumerate(class_names):
            mask = y_test == i
            if mask.sum() > 0:
                per_class[cn] = float(np.mean(preds[mask] == i))
            else:
                per_class[cn] = 0.0

        print(
            f"  Accuracy:       {accuracy:.4f} ({np.sum(preds == y_test)}/{len(y_test)})"
        )
        print(
            f"  Inference time: {avg_time * 1000:.1f} ms (avg over {TIMING_RUNS} runs)"
        )
        print(f"  Weight size:    {weight_size} bytes")
        print(f"  Per-class accuracy:")
        for cn, ca in per_class.items():
            print(f"    {cn:>12s}: {ca:.4f}")
        print()

        result = {
            "variant": variant,
            "accuracy": accuracy,
            "inference_time_s": avg_time,
            "weight_size_bytes": weight_size,
            "per_class_accuracy": per_class,
            "total_params": int(model.count_params()),
        }
        results.append(result)

        # Export .mem files for non-baseline variants
        if variant == "fp16":
            _write_mem_fp16(original_weights, quant_dir / "fp16")
        elif variant in ("int16", "int8"):
            n_bits = int(variant.replace("int", ""))
            _write_mem_int(original_weights, n_bits, quant_dir / variant)

    # ---- Restore original weights ----
    _set_weights_from_dict(model, original_weights)

    # ---- Save results JSON ----
    results_path = quant_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"model_type": model_type, "results": results}, f, indent=2)
    print(f"[quantize] Results saved to {results_path}")

    # ---- Generate comparison plot ----
    _generate_plots(results, model_type, quant_dir / "quantization_results.png")

    # ---- Print summary table ----
    print()
    print(f"{'=' * 70}")
    print(f"  QUANTIZATION COMPARISON SUMMARY — {model_type.upper()}")
    print(f"{'=' * 70}")
    print(
        f"  {'Variant':<10} {'Accuracy':>10} {'Time (ms)':>12} {'Size (B)':>10} {'Acc Drop':>10}"
    )
    print(f"  {'-' * 58}")

    baseline_acc = results[0]["accuracy"]
    for r in results:
        drop = baseline_acc - r["accuracy"]
        drop_str = f"{drop:+.4f}" if drop != 0 else "baseline"
        print(
            f"  {r['variant']:<10} {r['accuracy']:>10.4f} "
            f"{r['inference_time_s'] * 1000:>12.1f} "
            f"{r['weight_size_bytes']:>10d} "
            f"{drop_str:>10}"
        )

    print(f"{'=' * 70}")
    print()
    print(f"[quantize] Output files in {quant_dir}:")
    for p in sorted(quant_dir.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            rel = p.relative_to(adir)
            print(f"  {str(rel):50s} {size_kb:8.1f} KB")
    print()
    print("[quantize] Done.")


if __name__ == "__main__":
    main()
