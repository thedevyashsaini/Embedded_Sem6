"""
model.py — Build models for HAR: 1D-CNN, CNN+LSTM (DCLSTM), and WCLSTM.

Supported architectures (selected via `model_type` argument):

  1dcnn    — Lightweight 1D-CNN (19 features, 305 params).
             Conv1D(12, k=19, valid, relu) -> Flatten -> Dense(5, softmax)

  cnn_lstm — CNN + LSTM (DCLSTM): raw inertial signals (128, 9).
             Conv1D(64, k=5, relu) -> Conv1D(64, k=5, relu)
             -> LSTM(128) -> Dense(5, softmax)

  wclstm   — Wavelet-CNN + LSTM: wavelet-transformed signals.
             Conv1D(64, k=5, relu) -> Conv1D(64, k=5, relu)
             -> LSTM(128) -> Dense(5, softmax)
             (Input is wavelet-decomposed, so channel count differs.)

This module also provides `extract_model_spec()` which produces a
plain-Python dictionary describing every layer, shape, and parameter
count — used by export.py to write the JSON spec for the FPGA team.
"""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Valid model type identifiers
MODEL_TYPES = ("1dcnn", "cnn_lstm", "wclstm")


# ---------------------------------------------------------------------------
# 1D-CNN (original, 19-feature baseline)
# ---------------------------------------------------------------------------
def build_1dcnn(
    input_length: int = 19,
    conv_filters: int = 12,
    conv_kernel_size: int = 19,
    conv_stride: int = 1,
    conv_padding: str = "valid",
    conv_activation: str = "relu",
    num_classes: int = 5,
    output_activation: str = "softmax",
) -> keras.Model:
    """Construct the lightweight 1D-CNN (same as original)."""
    inp = keras.Input(shape=(input_length, 1), name="input")

    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=conv_kernel_size,
        strides=conv_stride,
        padding=conv_padding,
        activation=conv_activation,
        name="conv1d",
    )(inp)

    x = layers.Flatten(name="flatten")(x)

    out = layers.Dense(
        units=num_classes,
        activation=output_activation,
        name="dense_output",
    )(x)

    model = keras.Model(inputs=inp, outputs=out, name="har_1dcnn")
    return model


# ---------------------------------------------------------------------------
# CNN + LSTM (DCLSTM)
# ---------------------------------------------------------------------------
def build_cnn_lstm(
    input_timesteps: int = 128,
    input_channels: int = 9,
    conv1_filters: int = 64,
    conv1_kernel_size: int = 5,
    conv2_filters: int = 64,
    conv2_kernel_size: int = 5,
    conv_activation: str = "relu",
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    num_classes: int = 5,
    output_activation: str = "softmax",
) -> keras.Model:
    """Construct the CNN+LSTM (DCLSTM) model for raw inertial signals.

    Architecture:
        Input (128, 9)
        -> Conv1D(64, k=5, relu) -> Conv1D(64, k=5, relu)
        -> Dropout(0.3)
        -> LSTM(128)
        -> Dropout(0.3)
        -> Dense(5, softmax)
    """
    inp = keras.Input(shape=(input_timesteps, input_channels), name="input")

    x = layers.Conv1D(
        filters=conv1_filters,
        kernel_size=conv1_kernel_size,
        padding="same",
        activation=conv_activation,
        name="conv1d_1",
    )(inp)

    x = layers.Conv1D(
        filters=conv2_filters,
        kernel_size=conv2_kernel_size,
        padding="same",
        activation=conv_activation,
        name="conv1d_2",
    )(x)

    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.LSTM(lstm_units, name="lstm")(x)

    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    out = layers.Dense(
        units=num_classes,
        activation=output_activation,
        name="dense_output",
    )(x)

    model = keras.Model(inputs=inp, outputs=out, name="har_cnn_lstm")
    return model


# ---------------------------------------------------------------------------
# WCLSTM (Wavelet-CNN + LSTM)
# ---------------------------------------------------------------------------
def build_wclstm(
    input_timesteps: int = 66,
    input_channels: int = 9,
    conv1_filters: int = 64,
    conv1_kernel_size: int = 5,
    conv2_filters: int = 64,
    conv2_kernel_size: int = 5,
    conv_activation: str = "relu",
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    num_classes: int = 5,
    output_activation: str = "softmax",
) -> keras.Model:
    """Construct the WCLSTM model for wavelet-transformed signals.

    The input shape depends on the wavelet decomposition. With db4 level-2
    on 128-timestep signals, the approximation coefficients have length 36
    and detail coefficients lengths ~36 and ~66, concatenated to form a
    longer feature sequence per channel.

    Architecture is identical to DCLSTM but operates on wavelet features.

        Input (input_timesteps, input_channels)
        -> Conv1D(64, k=5, relu) -> Conv1D(64, k=5, relu)
        -> Dropout(0.3)
        -> LSTM(128)
        -> Dropout(0.3)
        -> Dense(5, softmax)
    """
    inp = keras.Input(shape=(input_timesteps, input_channels), name="input")

    x = layers.Conv1D(
        filters=conv1_filters,
        kernel_size=conv1_kernel_size,
        padding="same",
        activation=conv_activation,
        name="conv1d_1",
    )(inp)

    x = layers.Conv1D(
        filters=conv2_filters,
        kernel_size=conv2_kernel_size,
        padding="same",
        activation=conv_activation,
        name="conv1d_2",
    )(x)

    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.LSTM(lstm_units, name="lstm")(x)

    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    out = layers.Dense(
        units=num_classes,
        activation=output_activation,
        name="dense_output",
    )(x)

    model = keras.Model(inputs=inp, outputs=out, name="har_wclstm")
    return model


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------
def build_model(model_type: str = "1dcnn", **kwargs) -> keras.Model:
    """Build a model by type name.

    Parameters
    ----------
    model_type : str
        One of "1dcnn", "cnn_lstm", "wclstm".
    **kwargs
        Forwarded to the specific builder function.
    """
    if model_type == "1dcnn":
        return build_1dcnn(**kwargs)
    elif model_type == "cnn_lstm":
        return build_cnn_lstm(**kwargs)
    elif model_type == "wclstm":
        return build_wclstm(**kwargs)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: {MODEL_TYPES}"
        )


# ---------------------------------------------------------------------------
# Spec extraction (works for any Keras sequential/functional model)
# ---------------------------------------------------------------------------
def extract_model_spec(model: keras.Model) -> dict:
    """Extract a JSON-serialisable specification of the model.

    Returns a dict with:
      - model_name
      - total_params
      - layers: list of dicts per layer with name, type, config,
        output_shape, param_count, weight_shapes
    """
    # Safely get model-level shapes
    try:
        input_shape = list(model.input_shape)
    except AttributeError:
        input_shape = list(model.inputs[0].shape)
    try:
        output_shape = list(model.output_shape)
    except AttributeError:
        output_shape = list(model.outputs[0].shape)

    spec: dict = {
        "model_name": model.name,
        "total_params": int(model.count_params()),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "layers": [],
    }

    for layer in model.layers:
        layer_cfg = layer.get_config()
        weight_shapes = {w.name: list(w.shape) for w in layer.weights}

        # Safely get output_shape — various TF versions differ
        try:
            out_shape = list(layer.output_shape)
            if not out_shape:
                raise AttributeError
        except (AttributeError, RuntimeError):
            try:
                out_shape = list(layer.output.shape)
            except Exception:
                out_shape = list(
                    layer.get_config().get(
                        "batch_shape", layer.get_config().get("shape", [])
                    )
                )

        layer_info = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": out_shape,
            "param_count": int(layer.count_params()),
            "weight_shapes": weight_shapes,
            "config": {},
        }

        # Attach the most important config keys per layer type
        if isinstance(layer, layers.Conv1D):
            layer_info["config"] = {
                "filters": layer_cfg["filters"],
                "kernel_size": layer_cfg["kernel_size"],
                "strides": layer_cfg["strides"],
                "padding": layer_cfg["padding"],
                "activation": layer_cfg["activation"],
                "use_bias": layer_cfg["use_bias"],
            }
        elif isinstance(layer, layers.Dense):
            layer_info["config"] = {
                "units": layer_cfg["units"],
                "activation": layer_cfg["activation"],
                "use_bias": layer_cfg["use_bias"],
            }
        elif isinstance(layer, layers.LSTM):
            layer_info["config"] = {
                "units": layer_cfg["units"],
                "activation": layer_cfg["activation"],
                "recurrent_activation": layer_cfg["recurrent_activation"],
                "use_bias": layer_cfg["use_bias"],
                "return_sequences": layer_cfg["return_sequences"],
            }
        elif isinstance(layer, layers.Dropout):
            layer_info["config"] = {
                "rate": layer_cfg["rate"],
            }

        spec["layers"].append(layer_info)

    return spec
