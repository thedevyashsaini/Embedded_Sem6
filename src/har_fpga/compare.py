"""
compare.py — Compare results across all trained HAR models.

Reads model_spec.json and training_history.json from each model's
artifact directory and produces a side-by-side comparison table
and an optional comparison plot.

Usage:
    uv run python -m har_fpga.compare
    uv run python -m har_fpga.compare --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from har_fpga.model import MODEL_TYPES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def _load_model_results(model_type: str) -> dict | None:
    """Load training results for a model type, or None if not trained."""
    spec_path = ARTIFACTS_ROOT / model_type / "model_spec.json"
    if not spec_path.exists():
        return None
    with open(spec_path, "r") as f:
        spec = json.load(f)

    # Merge training_info from training_history.json if spec lacks it
    if not spec.get("training_info"):
        hist_path = ARTIFACTS_ROOT / model_type / "training_history.json"
        if hist_path.exists():
            with open(hist_path, "r") as f:
                hist = json.load(f)
            spec["training_info"] = {
                "final_train_accuracy": hist.get("accuracy", [0])[-1],
                "final_test_accuracy": hist.get("val_accuracy", [0])[-1],
                "final_test_loss": hist.get("val_loss", [0])[-1],
                "training_time_seconds": hist.get("training_time_seconds", 0),
                "epochs": len(hist.get("accuracy", [])),
            }
    return spec


def _load_quant_results(model_type: str) -> list[dict] | None:
    """Load quantization results for a model type, or None if not run."""
    results_path = ARTIFACTS_ROOT / model_type / "quantization" / "results.json"
    if not results_path.exists():
        return None
    with open(results_path, "r") as f:
        data = json.load(f)
    # Handle both old format (list) and new format (dict with "results" key)
    if isinstance(data, list):
        return data
    return data.get("results", [])


def _generate_comparison_plot(all_results: dict[str, dict], out_path: Path) -> None:
    """Generate a comparison bar chart across models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(all_results.keys())
    train_accs = []
    test_accs = []
    params = []
    times = []

    for mt in models:
        info = all_results[mt].get("training_info", {})
        train_accs.append(info.get("final_train_accuracy", 0) * 100)
        test_accs.append(info.get("final_test_accuracy", 0) * 100)
        params.append(all_results[mt].get("total_params", 0))
        times.append(info.get("training_time_seconds", 0))

    colors = ["#2196F3", "#4CAF50", "#FF9800"][: len(models)]
    x_labels = [m.upper() for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "HAR Model Comparison",
        fontsize=14,
        fontweight="bold",
    )

    # Test accuracy
    ax = axes[0]
    bars = ax.bar(x_labels, test_accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, test_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Parameters
    ax = axes[1]
    bars = ax.bar(x_labels, params, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Total Parameters")
    ax.set_title("Model Size")
    for bar, val in zip(bars, params):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(params) * 0.02,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Training time
    ax = axes[2]
    bars = ax.bar(x_labels, times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Training Time")
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.02,
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] Plot saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare results across all trained HAR models."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison bar charts",
    )
    args = parser.parse_args()

    # ---- Collect results ----
    all_results: dict[str, dict] = {}
    for mt in MODEL_TYPES:
        result = _load_model_results(mt)
        if result:
            all_results[mt] = result

    if not all_results:
        print("[compare] No trained models found. Train at least one model first:")
        for mt in MODEL_TYPES:
            print(f"          uv run python -m har_fpga.train --model {mt}")
        return

    # ---- Print comparison table ----
    print()
    print(f"{'=' * 80}")
    print(f"  HAR MODEL COMPARISON")
    print(f"{'=' * 80}")
    print(
        f"  {'Model':<12} {'Params':>10} {'Train Acc':>12} {'Test Acc':>12} "
        f"{'Test Loss':>12} {'Time (s)':>10}"
    )
    print(f"  {'-' * 72}")

    for mt, spec in all_results.items():
        info = spec.get("training_info", {})
        print(
            f"  {mt.upper():<12} "
            f"{spec.get('total_params', '?'):>10} "
            f"{info.get('final_train_accuracy', 0):>11.4f} "
            f"{info.get('final_test_accuracy', 0):>11.4f} "
            f"{info.get('final_test_loss', 0):>11.4f} "
            f"{info.get('training_time_seconds', 0):>10.1f}"
        )

    print(f"{'=' * 80}")

    # ---- Print quantization comparison if available ----
    quant_available = {}
    for mt in all_results:
        qr = _load_quant_results(mt)
        if qr:
            quant_available[mt] = qr

    if quant_available:
        print()
        print(f"{'=' * 80}")
        print(f"  QUANTIZATION COMPARISON (FP32 vs INT8)")
        print(f"{'=' * 80}")
        print(
            f"  {'Model':<12} {'FP32 Acc':>12} {'INT8 Acc':>12} {'Acc Drop':>12} "
            f"{'FP32 Size':>12} {'INT8 Size':>12}"
        )
        print(f"  {'-' * 72}")

        for mt, qr in quant_available.items():
            fp32 = next((r for r in qr if r["variant"] == "fp32"), None)
            int8 = next((r for r in qr if r["variant"] == "int8"), None)
            if fp32 and int8:
                drop = fp32["accuracy"] - int8["accuracy"]
                print(
                    f"  {mt.upper():<12} "
                    f"{fp32['accuracy']:>11.4f} "
                    f"{int8['accuracy']:>11.4f} "
                    f"{drop:>+11.4f} "
                    f"{fp32['weight_size_bytes']:>10d} B "
                    f"{int8['weight_size_bytes']:>10d} B"
                )

        print(f"{'=' * 80}")

    # ---- Not-yet-trained models ----
    missing = [mt for mt in MODEL_TYPES if mt not in all_results]
    if missing:
        print()
        print("[compare] Models not yet trained:")
        for mt in missing:
            print(f"          uv run python -m har_fpga.train --model {mt}")

    # ---- Generate plot ----
    if args.plot and len(all_results) >= 2:
        plot_path = ARTIFACTS_ROOT / "comparison.png"
        _generate_comparison_plot(all_results, plot_path)
    elif args.plot and len(all_results) < 2:
        print("[compare] Need at least 2 trained models to generate comparison plot.")

    print()


if __name__ == "__main__":
    main()
