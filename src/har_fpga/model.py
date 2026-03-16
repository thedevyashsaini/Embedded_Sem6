"""
model.py — Build models for HAR: MLP, 1D-CNN, 2D-CNN, CNN+LSTM (DCLSTM), and WCLSTM.

Supported architectures (selected via `model_type` argument):

  mlp      — Multi-Layer Perceptron on 19 pre-extracted features.
             Dense(64, relu) -> Dropout(0.3) -> Dense(32, relu)
             -> Dropout(0.3) -> Dense(16, relu) -> Dropout(0.3)
             -> Dense(5, softmax)

  1dcnn    — Lightweight 1D-CNN (19 features).
             Conv1D(12, k=19, valid, relu) -> Conv1D(8, k=1, valid, relu)
             -> Flatten -> Dense(5, softmax)

  2dcnn    — 2D-CNN on raw inertial signals (128, 9, 1).
             Conv2D(16, 3x3, relu) -> MaxPool(2x1) -> Conv2D(32, 3x3, relu)
             -> GlobalAvgPool2D -> Dense(5, softmax)

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
MODEL_TYPES = ("mlp", "1dcnn", "2dcnn", "cnn_lstm", "wclstm")


# ---------------------------------------------------------------------------
# MLP (Multi-Layer Perceptron)
# ---------------------------------------------------------------------------
def build_mlp(
    input_length: int = 19,
    hidden1_units: int = 64,
    hidden2_units: int = 32,
    hidden3_units: int = 16,
    hidden_activation: str = "relu",
    dropout_rate: float = 0.3,
    num_classes: int = 5,
    output_activation: str = "softmax",
) -> keras.Model:
    """Construct a Multi-Layer Perceptron for 19-feature HAR.

    Architecture:
        Input (19,)
        -> Dense(64, relu) -> Dropout(0.3)
        -> Dense(32, relu) -> Dropout(0.3)
        -> Dense(16, relu) -> Dropout(0.3)
        -> Dense(5, softmax)
    """
    inp = keras.Input(shape=(input_length,), name="input")

    x = layers.Dense(
        units=hidden1_units,
        activation=hidden_activation,
        name="dense_1",
    )(inp)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Dense(
        units=hidden2_units,
        activation=hidden_activation,
        name="dense_2",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    x = layers.Dense(
        units=hidden3_units,
        activation=hidden_activation,
        name="dense_3",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_3")(x)

    out = layers.Dense(
        units=num_classes,
        activation=output_activation,
        name="dense_output",
    )(x)

    model = keras.Model(inputs=inp, outputs=out, name="har_mlp")
    return model


# ---------------------------------------------------------------------------
# 1D-CNN (original, 19-feature baseline)
# ---------------------------------------------------------------------------
def build_1dcnn(
    input_length: int = 19,
    conv1_filters: int = 12,
    conv1_kernel_size: int = 19,
    conv1_stride: int = 1,
    conv1_padding: str = "valid",
    conv2_filters: int = 8,
    conv2_kernel_size: int = 1,
    conv2_stride: int = 1,
    conv2_padding: str = "valid",
    conv_activation: str = "relu",
    num_classes: int = 5,
    output_activation: str = "softmax",
) -> keras.Model:
    """Construct the lightweight 1D-CNN with two conv layers.

    Architecture:
        Input (19, 1)
        -> Conv1D(12, k=19, valid, relu)  -- output: (1, 12)
        -> Conv1D(8, k=1, valid, relu)    -- output: (1, 8)
        -> Flatten                        -- output: (8,)
        -> Dense(5, softmax)

    The second Conv1D uses kernel_size=1 (pointwise convolution) to add
    a non-linear feature mixing step with minimal parameter overhead.
    """
    inp = keras.Input(shape=(input_length, 1), name="input")

    x = layers.Conv1D(
        filters=conv1_filters,
        kernel_size=conv1_kernel_size,
        strides=conv1_stride,
        padding=conv1_padding,
        activation=conv_activation,
        name="conv1d_1",
    )(inp)

    x = layers.Conv1D(
        filters=conv2_filters,
        kernel_size=conv2_kernel_size,
        strides=conv2_stride,
        padding=conv2_padding,
        activation=conv_activation,
        name="conv1d_2",
    )(x)

    x = layers.Flatten(name="flatten")(x)

    out = layers.Dense(
        units=num_classes,
        activation=output_activation,
        name="dense_output",
    )(x)

    model = keras.Model(inputs=inp, outputs=out, name="har_1dcnn")
    return model


# ---------------------------------------------------------------------------
# 2D-CNN (raw inertial signals treated as 2D image)
# ---------------------------------------------------------------------------
def build_2dcnn(
    input_timesteps: int = 128,
    input_channels: int = 9,
    conv1_filters: int = 16,
    conv1_kernel: tuple[int, int] = (3, 3),
    conv2_filters: int = 32,
    conv2_kernel: tuple[int, int] = (3, 3),
    conv_activation: str = "relu",
    dropout_rate: float = 0.3,
    num_classes: int = 5,
    output_activation: str = "softmax",
) -> keras.Model:
    """Construct a 2D-CNN for raw inertial signals.

    The raw signals (128 timesteps x 9 channels) are treated as a single-
    channel 2D image of shape (128, 9, 1). Two Conv2D layers with max-
    pooling extract spatial patterns, followed by global average pooling
    and a dense output layer.

    Architecture:
        Input (128, 9, 1)
        -> Conv2D(16, 3x3, same, relu) -> MaxPool2D(2x1)
        -> Conv2D(32, 3x3, same, relu) -> MaxPool2D(2x1)
        -> Dropout(0.3)
        -> GlobalAveragePooling2D
        -> Dense(5, softmax)
    """
    inp = keras.Input(shape=(input_timesteps, input_channels, 1), name="input")

    x = layers.Conv2D(
        filters=conv1_filters,
        kernel_size=conv1_kernel,
        padding="same",
        activation=conv_activation,
        name="conv2d_1",
    )(inp)
    x = layers.MaxPooling2D(pool_size=(2, 1), name="maxpool_1")(x)

    x = layers.Conv2D(
        filters=conv2_filters,
        kernel_size=conv2_kernel,
        padding="same",
        activation=conv_activation,
        name="conv2d_2",
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), name="maxpool_2")(x)

    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    out = layers.Dense(
        units=num_classes,
        activation=output_activation,
        name="dense_output",
    )(x)

    model = keras.Model(inputs=inp, outputs=out, name="har_2dcnn")
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
        One of "mlp", "1dcnn", "2dcnn", "cnn_lstm", "wclstm".
    **kwargs
        Forwarded to the specific builder function.
    """
    if model_type == "mlp":
        return build_mlp(**kwargs)
    elif model_type == "1dcnn":
        return build_1dcnn(**kwargs)
    elif model_type == "2dcnn":
        return build_2dcnn(**kwargs)
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
        elif isinstance(layer, layers.Conv2D):
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
        elif isinstance(layer, (layers.MaxPooling2D, layers.MaxPooling1D)):
            layer_info["config"] = {
                "pool_size": layer_cfg["pool_size"],
                "strides": layer_cfg["strides"],
                "padding": layer_cfg["padding"],
            }
        elif isinstance(
            layer, (layers.GlobalAveragePooling2D, layers.GlobalAveragePooling1D)
        ):
            layer_info["config"] = {}

        spec["layers"].append(layer_info)

    return spec
