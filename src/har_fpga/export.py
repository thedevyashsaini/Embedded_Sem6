"""
export.py — Export model weights and specification for FPGA deployment.

Produces:
  1. model_spec.json  — Layer-by-layer architecture, shapes, param counts,
                         activations, feature names, class names.
  2. model_weights.mem — All weights and biases in a flat text format that
                          can be read directly by Verilog $readmemh or
                          parsed by an HLS tool.

.mem format details:
  - One floating-point value per line (as decimal string).
  - Sections are separated by comment lines starting with //.
  - Each section header names the weight tensor and its shape.
  - Weight order within each tensor follows row-major (C) order,
    matching numpy's default flatten().

Usage:
    uv run python -m har_fpga.export
    uv run python -m har_fpga.export --model artifacts/har_model.keras
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from har_fpga.model import extract_model_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def _float_to_hex(value: float) -> str:
    """Convert a float32 to its IEEE-754 hex representation (8 hex digits)."""
    return format(struct.unpack(">I", struct.pack(">f", value))[0], "08x")


def export_weights_mem(model: keras.Model, path: Path) -> None:
    """Export all model weights to a .mem file.

    Format per weight tensor:
        // <tensor_name> shape=(<dims>) count=<N>
        <hex_value_1>
        <hex_value_2>
        ...

    Values are IEEE-754 float32 encoded as 8-char hex strings,
    suitable for Verilog $readmemh.
    """
    lines: list[str] = []
    lines.append("// HAR 1D-CNN Model Weights — FPGA .mem export")
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
    """Export model specification JSON (re-extract from model)."""
    spec = extract_model_spec(model)
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[export] Model spec saved to {path}")


def export_weights_readable(model: keras.Model, path: Path) -> None:
    """Export a human-readable summary of all weights (decimal values)."""
    lines: list[str] = []
    lines.append("HAR 1D-CNN Model Weights — Human-Readable Export")
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
        default=str(ARTIFACT_DIR / "har_model.keras"),
        help="Path to saved Keras model",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[export] ERROR: Model file not found at {model_path}")
        print("         Run training first:  uv run python -m har_fpga.train")
        raise SystemExit(1)

    print(f"[export] Loading model from {model_path} ...")
    model = keras.models.load_model(model_path)
    model.summary()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    export_spec_json(model, ARTIFACT_DIR / "model_spec.json")
    export_weights_mem(model, ARTIFACT_DIR / "model_weights.mem")
    export_weights_readable(model, ARTIFACT_DIR / "weights_readable.txt")

    print("\n[export] Done. Files in artifacts/:")
    for p in sorted(ARTIFACT_DIR.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"         {p.name:30s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
