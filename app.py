"""
app.py — Streamlit dashboard for HAR model comparison.

Launch:
    uv run streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
MODEL_TYPES = ["mlp", "1dcnn", "2dcnn", "cnn_lstm", "wclstm"]

MODEL_DISPLAY = {
    "mlp": "MLP",
    "1dcnn": "1D-CNN",
    "2dcnn": "2D-CNN",
    "cnn_lstm": "CNN+LSTM",
    "wclstm": "WCLSTM",
}

MODEL_COLORS = {
    "MLP": "#AB47BC",
    "1D-CNN": "#2196F3",
    "2D-CNN": "#E91E63",
    "CNN+LSTM": "#4CAF50",
    "WCLSTM": "#FF9800",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_all_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load model specs and quantization results into DataFrames."""
    rows_spec = []
    rows_quant = []

    for mt in MODEL_TYPES:
        spec_path = ARTIFACTS_ROOT / mt / "model_spec.json"
        if not spec_path.exists():
            continue
        with open(spec_path) as f:
            spec = json.load(f)

        # Try training_info from spec, fall back to training_history.json
        info = spec.get("training_info", {})
        if not info:
            hist_path = ARTIFACTS_ROOT / mt / "training_history.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    hist = json.load(f)
                info = {
                    "final_train_accuracy": hist.get("accuracy", [0])[-1],
                    "final_test_accuracy": hist.get("val_accuracy", [0])[-1],
                    "final_test_loss": hist.get("val_loss", [0])[-1],
                    "training_time_seconds": hist.get("training_time_seconds", 0),
                }

        rows_spec.append(
            {
                "model_id": mt,
                "Model": MODEL_DISPLAY.get(mt, mt.upper()),
                "Parameters": spec.get("total_params", 0),
                "Train Acc (%)": round(info.get("final_train_accuracy", 0) * 100, 2),
                "Test Acc (%)": round(info.get("final_test_accuracy", 0) * 100, 2),
                "Test Loss": round(info.get("final_test_loss", 0), 4),
                "Train Time (s)": round(info.get("training_time_seconds", 0), 1),
            }
        )

        # Quantization results
        qr_path = ARTIFACTS_ROOT / mt / "quantization" / "results.json"
        if qr_path.exists():
            with open(qr_path) as f:
                qdata = json.load(f)
            results = qdata.get("results", qdata) if isinstance(qdata, dict) else qdata
            for r in results:
                rows_quant.append(
                    {
                        "model_id": mt,
                        "Model": MODEL_DISPLAY.get(mt, mt.upper()),
                        "Variant": r["variant"].upper(),
                        "Accuracy (%)": round(r["accuracy"] * 100, 2),
                        "Inference Time (ms)": round(r["inference_time_s"] * 1000, 1),
                        "Weight Size (B)": r["weight_size_bytes"],
                        "Parameters": r.get("total_params", 0),
                    }
                )

    df_spec = pd.DataFrame(rows_spec)
    df_quant = pd.DataFrame(rows_quant)
    return df_spec, df_quant


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HAR Model Comparison",
    page_icon="🏃",
    layout="wide",
)

st.title("HAR Model Comparison Dashboard")
st.markdown(
    "Interactive comparison of **MLP**, **1D-CNN**, **2D-CNN**, **CNN+LSTM (DCLSTM)**, "
    "and **WCLSTM** architectures for Human Activity Recognition on the UCI HAR dataset."
)

df_spec, df_quant = load_all_results()

if df_spec.empty:
    st.error("No trained models found. Train at least one model first.")
    st.stop()

# ---------------------------------------------------------------------------
# Section 1: Overview metrics
# ---------------------------------------------------------------------------
st.header("Model Overview")

cols = st.columns(len(df_spec))
for i, (_, row) in enumerate(df_spec.iterrows()):
    with cols[i]:
        st.metric(label=row["Model"], value=f"{row['Test Acc (%)']:.2f}%")
        st.caption(f"{row['Parameters']:,} params")

st.dataframe(
    df_spec[
        [
            "Model",
            "Parameters",
            "Train Acc (%)",
            "Test Acc (%)",
            "Test Loss",
            "Train Time (s)",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Section 2: 3D Comparison (Accuracy vs Inference Time vs Parameters)
# ---------------------------------------------------------------------------
st.header("3D Model Comparison")
st.markdown(
    "Each point represents a model at **FP32** precision. "
    "Axes: **Test Accuracy**, **Inference Time**, and **Parameter Count**. "
    "Drag to rotate, scroll to zoom."
)

if not df_quant.empty:
    # Use FP32 variant for the 3D plot
    df_fp32 = df_quant[df_quant["Variant"] == "FP32"].copy()

    if not df_fp32.empty:
        fig_3d = px.scatter_3d(
            df_fp32,
            x="Accuracy (%)",
            y="Inference Time (ms)",
            z="Parameters",
            color="Model",
            color_discrete_map=MODEL_COLORS,
            text="Model",
            size_max=20,
            title="Accuracy vs Inference Time vs Model Size (FP32)",
        )

        fig_3d.update_traces(
            marker=dict(size=14, line=dict(width=2, color="DarkSlateGrey")),
            textposition="top center",
            textfont=dict(size=12, color="white"),
        )

        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Test Accuracy (%)",
                yaxis_title="Inference Time (ms)",
                zaxis_title="Parameters",
                bgcolor="rgba(0,0,0,0)",
            ),
            height=650,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Bar charts (Accuracy + Inference Time side by side)
# ---------------------------------------------------------------------------
st.header("Model Comparison Charts")

if not df_quant.empty:
    df_fp32 = df_quant[df_quant["Variant"] == "FP32"].copy()

    col1, col2 = st.columns(2)

    with col1:
        fig_acc = px.bar(
            df_fp32,
            x="Model",
            y="Accuracy (%)",
            color="Model",
            color_discrete_map=MODEL_COLORS,
            title="Test Accuracy (FP32)",
            text="Accuracy (%)",
        )
        fig_acc.update_traces(textposition="outside", texttemplate="%{text:.2f}%")
        fig_acc.update_layout(yaxis_range=[0, 105], showlegend=False, height=450)
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        fig_time = px.bar(
            df_fp32,
            x="Model",
            y="Inference Time (ms)",
            color="Model",
            color_discrete_map=MODEL_COLORS,
            title="Inference Time (FP32, full test set)",
            text="Inference Time (ms)",
        )
        fig_time.update_traces(textposition="outside", texttemplate="%{text:.1f} ms")
        fig_time.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig_time, use_container_width=True)

    # Parameter count
    fig_params = px.bar(
        df_fp32,
        x="Model",
        y="Parameters",
        color="Model",
        color_discrete_map=MODEL_COLORS,
        title="Model Size (Parameter Count)",
        text="Parameters",
        log_y=True,
    )
    fig_params.update_traces(textposition="outside", texttemplate="%{text:,}")
    fig_params.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_params, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Per-class accuracy heatmap
# ---------------------------------------------------------------------------
st.header("Per-Class Accuracy Heatmap")

CLASS_NAMES = ["WALKING", "SITTING", "STANDING", "LAYING"]

if not df_quant.empty:
    df_fp32_detail = df_quant[df_quant["Variant"] == "FP32"].copy()
    # Load per-class from quantization results
    heatmap_data = []
    for mt in MODEL_TYPES:
        qr_path = ARTIFACTS_ROOT / mt / "quantization" / "results.json"
        if not qr_path.exists():
            continue
        with open(qr_path) as f:
            qdata = json.load(f)
        results = qdata.get("results", qdata) if isinstance(qdata, dict) else qdata
        fp32_result = next((r for r in results if r["variant"] == "fp32"), None)
        if fp32_result and "per_class_accuracy" in fp32_result:
            for cn in CLASS_NAMES:
                acc = fp32_result["per_class_accuracy"].get(cn, 0)
                heatmap_data.append(
                    {
                        "Model": MODEL_DISPLAY.get(mt, mt.upper()),
                        "Class": cn,
                        "Accuracy (%)": round(acc * 100, 2),
                    }
                )

    if heatmap_data:
        df_heat = pd.DataFrame(heatmap_data)
        pivot = df_heat.pivot(index="Class", columns="Model", values="Accuracy (%)")
        # Reorder columns to match MODEL_TYPES order
        ordered_cols = [
            MODEL_DISPLAY[mt]
            for mt in MODEL_TYPES
            if MODEL_DISPLAY[mt] in pivot.columns
        ]
        pivot = pivot[ordered_cols]

        fig_heat = px.imshow(
            pivot,
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            title="Per-Class Test Accuracy (%) by Model",
            aspect="auto",
        )
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5: Quantization comparison
# ---------------------------------------------------------------------------
st.header("Quantization Impact")

if not df_quant.empty:
    st.markdown(
        "Comparison of accuracy across quantization variants (FP32, FP16, INT16, INT8) for each model."
    )

    fig_quant = px.bar(
        df_quant,
        x="Model",
        y="Accuracy (%)",
        color="Variant",
        barmode="group",
        title="Accuracy by Quantization Variant",
        text="Accuracy (%)",
        color_discrete_sequence=["#2196F3", "#4CAF50", "#FF9800", "#F44336"],
    )
    fig_quant.update_traces(textposition="outside", texttemplate="%{text:.2f}%")
    fig_quant.update_layout(yaxis_range=[0, 105], height=450)
    st.plotly_chart(fig_quant, use_container_width=True)

    # Weight size comparison
    fig_wsize = px.bar(
        df_quant,
        x="Model",
        y="Weight Size (B)",
        color="Variant",
        barmode="group",
        title="Weight Size by Quantization Variant",
        text="Weight Size (B)",
        color_discrete_sequence=["#2196F3", "#4CAF50", "#FF9800", "#F44336"],
        log_y=True,
    )
    fig_wsize.update_traces(textposition="outside", texttemplate="%{text:,} B")
    fig_wsize.update_layout(height=450)
    st.plotly_chart(fig_wsize, use_container_width=True)

    # Full quantization table
    st.subheader("Full Quantization Results")
    st.dataframe(
        df_quant[
            [
                "Model",
                "Variant",
                "Accuracy (%)",
                "Inference Time (ms)",
                "Weight Size (B)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Section 6: Existing plots from artifacts
# ---------------------------------------------------------------------------
st.header("Training & Quantization Plots")

col_a, col_b = st.columns(2)

comparison_plot = ARTIFACTS_ROOT / "comparison.png"
if comparison_plot.exists():
    with col_a:
        st.image(
            str(comparison_plot),
            caption="Cross-Model Comparison",
            use_container_width=True,
        )

# Show per-model quantization plots
for mt in MODEL_TYPES:
    qplot = ARTIFACTS_ROOT / mt / "quantization" / "quantization_results.png"
    if qplot.exists():
        with col_b:
            st.image(
                str(qplot),
                caption=f"{MODEL_DISPLAY.get(mt, mt.upper())} Quantization",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "HAR-FPGA: Human Activity Recognition for FPGA Deployment | "
    "UCI HAR Dataset | TensorFlow + Keras"
)
