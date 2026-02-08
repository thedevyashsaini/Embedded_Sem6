"""
model.py — Build the lightweight 1D-CNN for HAR.

Architecture (FPGA-friendly, matches the research paper):
  Input (19, 1)
    -> Conv1D(filters=12, kernel_size=19, stride=1, padding='valid', activation='relu')
    -> Flatten
    -> Dense(5, activation='softmax')

No pooling, no dropout, no batch-norm — intentionally minimal so the
hardware team can translate this directly to RTL / HLS.

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


def build_model(
    input_length: int = 19,
    conv_filters: int = 12,
    conv_kernel_size: int = 19,
    conv_stride: int = 1,
    conv_padding: str = "valid",
    conv_activation: str = "relu",
    num_classes: int = 5,
    output_activation: str = "softmax",
) -> keras.Model:
    """Construct and return the (uncompiled) Keras model.

    Parameters mirror configs/training.json["model"] so callers can
    simply unpack the config dict:

        model = build_model(**cfg["model"])
    """
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

        spec["layers"].append(layer_info)

    return spec
