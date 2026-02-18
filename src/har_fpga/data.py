"""
data.py — Download, extract, load, and prepare UCI HAR dataset.

Supports two data modes:
  1. "features" — 19 pre-extracted statistical features (for 1D-CNN).
  2. "raw"      — 128-timestep x 9-channel raw inertial signals
                  (for CNN+LSTM and WCLSTM).

Responsibilities:
  1. Download UCI HAR Dataset zip if not already cached.
  2. Extract the zip to data/.
  3. Load pre-extracted feature vectors OR raw inertial signals.
  4. Load labels (y_train.txt, y_test.txt).
  5. Select the 19 features (features mode) or stack 9 signal channels (raw mode).
  6. Remap labels to 5 classes per configs/training.json.

This module is pure data I/O — no model or preprocessing logic.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00240/UCI%20HAR%20Dataset.zip"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"
ZIP_PATH = DATA_DIR / "UCI_HAR_Dataset.zip"
EXTRACT_DIR = DATA_DIR / "UCI HAR Dataset"

# The 9 raw inertial signal file basenames (order matters for channel dim)
_SIGNAL_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


# ---------------------------------------------------------------------------
# Download & extract
# ---------------------------------------------------------------------------
def download_dataset(force: bool = False) -> Path:
    """Download the UCI HAR zip file into data/. Returns path to zip."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        print(f"[data] Zip already cached at {ZIP_PATH}")
        return ZIP_PATH

    print(f"[data] Downloading dataset from {DATASET_URL} ...")
    resp = requests.get(DATASET_URL, stream=True, timeout=120)
    resp.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    print(f"[data] Saved to {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
    return ZIP_PATH


def extract_dataset(force: bool = False) -> Path:
    """Extract the zip. Returns path to extracted root directory."""
    if EXTRACT_DIR.exists() and not force:
        print(f"[data] Already extracted at {EXTRACT_DIR}")
        return EXTRACT_DIR

    print("[data] Extracting zip ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"[data] Extracted to {EXTRACT_DIR}")
    return EXTRACT_DIR


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------
def _load_feature_config() -> tuple[list[int], list[str]]:
    """Return (0-based indices, feature names) from configs/features.json."""
    cfg_path = CONFIG_DIR / "features.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    # Convert 1-based indices to 0-based
    indices_0 = [i - 1 for i in cfg["feature_indices_1based"]]
    return indices_0, cfg["feature_names"]


def _load_training_config() -> dict:
    """Load configs/training.json."""
    cfg_path = CONFIG_DIR / "training.json"
    with open(cfg_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Label remapping
# ---------------------------------------------------------------------------
def remap_labels(y: np.ndarray, label_map: dict[str, int]) -> np.ndarray:
    """Remap original UCI labels (1-6) to merged classes (0-4).

    label_map keys are string representations of original labels,
    values are new integer class ids.

    Any label NOT in the map is assigned to class 4 (TRANSITION).
    """
    y_new = np.full_like(y, fill_value=4)  # default = TRANSITION
    for orig_str, new_id in label_map.items():
        y_new[y == int(orig_str)] = new_id
    return y_new


# ---------------------------------------------------------------------------
# Raw inertial signal loading
# ---------------------------------------------------------------------------
def _load_inertial_signals(split_dir: Path) -> np.ndarray:
    """Load all 9 inertial signal files from a split directory.

    Each file has shape (N, 128) — 128 timesteps per sample.
    Stacks them into (N, 128, 9).

    Parameters
    ----------
    split_dir : Path
        e.g. data/UCI HAR Dataset/train/Inertial Signals/

    Returns
    -------
    signals : ndarray of shape (N, 128, 9)
    """
    inertial_dir = split_dir / "Inertial Signals"
    suffix = "train" if "train" in split_dir.name else "test"
    channels = []
    for sig_name in _SIGNAL_FILES:
        fpath = inertial_dir / f"{sig_name}_{suffix}.txt"
        arr = np.loadtxt(fpath)  # (N, 128)
        channels.append(arr)
    # Stack along last axis: list of (N,128) -> (N, 128, 9)
    return np.stack(channels, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main loading functions
# ---------------------------------------------------------------------------
def load_har_data(
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load and prepare the HAR dataset (19 pre-extracted features mode).

    Returns
    -------
    X_train : ndarray of shape (N_train, 19)
    y_train : ndarray of shape (N_train,)   — labels in {0,1,2,3,4}
    X_test  : ndarray of shape (N_test, 19)
    y_test  : ndarray of shape (N_test,)
    feature_names : list[str]  — the 19 selected feature names
    class_names   : list[str]  — ["WALKING","SITTING","STANDING","LAYING","TRANSITION"]
    """
    if download:
        download_dataset()
        extract_dataset()

    # --- Load full feature matrices (561 columns) ---
    train_dir = EXTRACT_DIR / "train"
    test_dir = EXTRACT_DIR / "test"

    X_train_full = np.loadtxt(train_dir / "X_train.txt")
    X_test_full = np.loadtxt(test_dir / "X_test.txt")
    y_train_raw = np.loadtxt(train_dir / "y_train.txt", dtype=int)
    y_test_raw = np.loadtxt(test_dir / "y_test.txt", dtype=int)

    print(f"[data] Loaded X_train: {X_train_full.shape}, X_test: {X_test_full.shape}")

    # --- Select 19 features ---
    feat_idx, feat_names = _load_feature_config()
    X_train = X_train_full[:, feat_idx]
    X_test = X_test_full[:, feat_idx]
    print(f"[data] Selected {len(feat_idx)} features -> X_train: {X_train.shape}")

    # --- Remap labels ---
    cfg = _load_training_config()
    label_map = cfg["label_map"]
    class_names = cfg["class_names"]

    y_train = remap_labels(y_train_raw, label_map)
    y_test = remap_labels(y_test_raw, label_map)

    unique, counts = np.unique(y_train, return_counts=True)
    print("[data] Training label distribution:")
    for u, c in zip(unique, counts):
        print(f"       {class_names[u]:>12s} (class {u}): {c}")

    return X_train, y_train, X_test, y_test, feat_names, class_names


def load_har_raw(
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load and prepare the HAR dataset (raw inertial signals mode).

    Returns
    -------
    X_train : ndarray of shape (N_train, 128, 9)
    y_train : ndarray of shape (N_train,)   — labels in {0,1,2,3,4}
    X_test  : ndarray of shape (N_test, 128, 9)
    y_test  : ndarray of shape (N_test,)
    signal_names : list[str]  — the 9 signal channel names
    class_names  : list[str]  — ["WALKING","SITTING","STANDING","LAYING","TRANSITION"]
    """
    if download:
        download_dataset()
        extract_dataset()

    train_dir = EXTRACT_DIR / "train"
    test_dir = EXTRACT_DIR / "test"

    print("[data] Loading raw inertial signals (9 channels x 128 timesteps) ...")
    X_train = _load_inertial_signals(train_dir)
    X_test = _load_inertial_signals(test_dir)

    y_train_raw = np.loadtxt(train_dir / "y_train.txt", dtype=int)
    y_test_raw = np.loadtxt(test_dir / "y_test.txt", dtype=int)

    print(f"[data] Loaded X_train: {X_train.shape}, X_test: {X_test.shape}")

    # --- Remap labels ---
    cfg = _load_training_config()
    label_map = cfg["label_map"]
    class_names = cfg["class_names"]

    y_train = remap_labels(y_train_raw, label_map)
    y_test = remap_labels(y_test_raw, label_map)

    unique, counts = np.unique(y_train, return_counts=True)
    print("[data] Training label distribution:")
    for u, c in zip(unique, counts):
        print(f"       {class_names[u]:>12s} (class {u}): {c}")

    return X_train, y_train, X_test, y_test, _SIGNAL_FILES, class_names
