"""
preprocess.py — Z-score normalization utilities.

Responsibilities:
  1. Fit scaler statistics (mean, std) on training data.
  2. Transform data using fitted statistics.
  3. Save / load scaler to JSON (for reproducibility and FPGA deployment).

The hardware team needs these exact mean/std values to replicate
normalization in fixed-point on the FPGA.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ZScoreScaler:
    """Simple z-score (standard) scaler: x' = (x - mean) / std.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
    std_  : ndarray of shape (n_features,)
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        """Compute per-feature mean and std from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        self.mean_ = X.mean(axis=0).astype(np.float64)
        self.std_ = X.std(axis=0).astype(np.float64)
        # Guard against zero std (constant features)
        self.std_[self.std_ == 0.0] = 1.0
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_normalized : ndarray of same shape, dtype float32
        """
        assert self.mean_ is not None, "Scaler not fitted. Call fit() first."
        return ((X - self.mean_) / self.std_).astype(np.float32)

    # ------------------------------------------------------------------
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save scaler statistics to a JSON file.

        The JSON contains:
          - mean: list of floats (one per feature)
          - std:  list of floats (one per feature)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[preprocess] Scaler saved to {path}")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "ZScoreScaler":
        """Load scaler statistics from a JSON file."""
        with open(path, "r") as f:
            payload = json.load(f)
        scaler = cls()
        scaler.mean_ = np.array(payload["mean"], dtype=np.float64)
        scaler.std_ = np.array(payload["std"], dtype=np.float64)
        return scaler
