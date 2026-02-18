"""
export.py — Export model weights and specification for FPGA deployment.

Supports all three architectures: 1dcnn, cnn_lstm, wclstm.

Produces (in artifacts/<model_type>/):
  1. model_spec.json  — Layer-by-layer architecture, shapes, param counts.
  2. model_weights.mem — All weights in IEEE-754 hex for Verilog $readmemh.
  3. weights_readable.txt — Human-readable weight dump.

Usage:
    uv run python -m har_fpga.export --model 1dcnn
    uv run python -m har_fpga.export --model cnn_lstm
    uv run python -m har_fpga.export --model wclstm
    uv run python -m har_fpga.export --model 1dcnn --model-path path/to/model.keras
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from har_fpga.model import extract_model_spec, MODEL_TYPES

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _artifact_dir(model_type: str) -> Path:
    return PROJECT_ROOT / "artifacts" / model_type


def _float_to_hex(value: float) -> str:
    """Convert a float32 to its IEEE-754 hex representation (8 hex digits)."""
    return format(struct.unpack(">I", struct.pack(">f", value))[0], "08x")


def export_weights_mem(model: keras.Model, path: Path) -> None:
    """Export all model weights to a .mem file.

    Values are IEEE-754 float32 encoded as 8-char hex strings,
    suitable for Verilog $readmemh.
    """
    lines: list[str] = []
    lines.append(f"// {model.name} Model Weights — FPGA .mem export")
    lines.append(f"// Total parameters: {model.count_params()}")
    lines.append("")

    for layer in model.layers:
        for w in layer.weights:
            arr = w.numpy().astype(np.float32).flatten()
            shape_str = "x".join(str(d) for d in w.shape)
            lines.append(f"// {w.name}  shape=({shape_str})  count={arr.size}")
            for val in arr:
                lines.append(_float_to_hex(float(val)))
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[export] Weights .mem file saved to {path}")
    print(f"         Total lines (including comments): {len(lines)}")


def export_spec_json(model: keras.Model, path: Path) -> None:
    """Export model specification JSON (re-extract from model).

    Preserves existing training_info if the file already exists
    (written by train.py) so that compare.py can read it.
    """
    # Load existing spec to preserve training_info
    existing_training_info = None
    if path.exists():
        with open(path, "r") as f:
            existing = json.load(f)
        existing_training_info = existing.get("training_info")

    spec = extract_model_spec(model)
    if existing_training_info:
        spec["training_info"] = existing_training_info

    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[export] Model spec saved to {path}")


def export_weights_readable(model: keras.Model, path: Path) -> None:
    """Export a human-readable summary of all weights (decimal values)."""
    lines: list[str] = []
    lines.append(f"{model.name} Model Weights — Human-Readable Export")
    lines.append(f"Total parameters: {model.count_params()}")
    lines.append("=" * 60)

    for layer in model.layers:
        if not layer.weights:
            continue
        lines.append(f"\nLayer: {layer.name} ({layer.__class__.__name__})")
        lines.append("-" * 40)
        for w in layer.weights:
            arr = w.numpy()
            lines.append(f"  {w.name}  shape={list(w.shape)}")
            # Print small tensors inline, larger ones summarised
            if arr.size <= 100:
                flat = arr.flatten()
                for i in range(0, len(flat), 8):
                    chunk = flat[i : i + 8]
                    lines.append("    " + "  ".join(f"{v:+.6f}" for v in chunk))
            else:
                lines.append(
                    f"    min={arr.min():.6f}  max={arr.max():.6f}  "
                    f"mean={arr.mean():.6f}  std={arr.std():.6f}"
                )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[export] Readable weights summary saved to {path}")


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained HAR model to .mem and JSON spec for FPGA."
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
        help="Path to saved Keras model (overrides default artifacts/<model>/har_model.keras)",
    )
    args = parser.parse_args()

    model_type = args.model
    adir = _artifact_dir(model_type)
    model_path = Path(args.model_path) if args.model_path else adir / "har_model.keras"

    if not model_path.exists():
        print(f"[export] ERROR: Model file not found at {model_path}")
        print(
            f"         Run training first:  uv run python -m har_fpga.train --model {model_type}"
        )
        raise SystemExit(1)

    print(f"[export] Model type: {model_type.upper()}")
    print(f"[export] Loading model from {model_path} ...")
    model = keras.models.load_model(model_path)
    model.summary()

    adir.mkdir(parents=True, exist_ok=True)
    export_spec_json(model, adir / "model_spec.json")
    export_weights_mem(model, adir / "model_weights.mem")
    export_weights_readable(model, adir / "weights_readable.txt")

    print(f"\n[export] Done. Files in {adir}:")
    for p in sorted(adir.iterdir()):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"         {p.name:30s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
