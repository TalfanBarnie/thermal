"""Background model inference from a .npz artefact. numpy only.

Artefact layout (all arrays float32/float64 unless noted):
  meta            JSON string: {"model_type": "mlp"|"ridge_poly", "feature_names": [...],
                                "target": ..., "sat": ..., "platform": ..., ...}
  x_mean, x_std   feature standardisation (n_features,)
  y_mean, y_std   target standardisation (scalars)
  mlp:            W0, b0, W1, b1, ... (last layer has 2 outputs: standardised mean, log-variance)
  ridge_poly:     powers (n_terms, n_features) int, w_mean (n_terms,), b_mean, w_logvar, b_logvar
Outputs are de-standardised: mean and standard deviation in target units.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

LOGVAR_MIN, LOGVAR_MAX = -12.0, 6.0


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def poly_features(xs: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Monomial expansion: column j is prod_i xs[:, i] ** powers[j, i]."""
    out = np.ones((xs.shape[0], powers.shape[0]), dtype=np.float64)
    for j, p in enumerate(powers):
        for i, e in enumerate(p):
            if e:
                out[:, j] *= xs[:, i] ** e
    return out


class BackgroundModel:
    def __init__(self, arrays: dict[str, Any]):
        self.meta = json.loads(str(arrays["meta"]))
        self.model_type = self.meta["model_type"]
        self.feature_names = list(self.meta["feature_names"])
        self.x_mean = np.asarray(arrays["x_mean"], dtype=np.float64)
        self.x_std = np.asarray(arrays["x_std"], dtype=np.float64)
        self.y_mean = float(arrays["y_mean"])
        self.y_std = float(arrays["y_std"])
        self.arrays = arrays

    @classmethod
    def load(cls, path: Path) -> BackgroundModel:
        with np.load(path, allow_pickle=False) as npz:
            return cls({k: npz[k] for k in npz.files})

    def _forward_std(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model_type == "mlp":
            h = xs
            n_layers = self.meta["n_layers"]
            for i in range(n_layers):
                h = h @ np.asarray(self.arrays[f"W{i}"], dtype=np.float64) + np.asarray(
                    self.arrays[f"b{i}"], dtype=np.float64
                )
                if i < n_layers - 1:
                    h = gelu_tanh(h)
            return h[:, 0], h[:, 1]
        if self.model_type == "ridge_poly":
            p = poly_features(xs, np.asarray(self.arrays["powers"]))
            mean = p @ np.asarray(self.arrays["w_mean"], dtype=np.float64) + float(
                self.arrays["b_mean"]
            )
            logvar = p @ np.asarray(self.arrays["w_logvar"], dtype=np.float64) + float(
                self.arrays["b_logvar"]
            )
            return mean, logvar
        raise ValueError(f"unknown model type {self.model_type}")

    def predict(self, X: np.ndarray, feature_names: list[str] | None = None):
        """Return (mean, std) in target units for a feature matrix (n, n_features)."""
        if feature_names is not None and feature_names != self.feature_names:
            raise ValueError("feature order mismatch between cutout and model")
        xs = (np.asarray(X, dtype=np.float64) - self.x_mean) / self.x_std
        mean_s, logvar_s = self._forward_std(xs)
        logvar_s = np.clip(logvar_s, LOGVAR_MIN, LOGVAR_MAX)
        mean = mean_s * self.y_std + self.y_mean
        std = np.exp(0.5 * logvar_s) * self.y_std
        return mean, std


def save_model(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        k: (np.asarray(v) if k != "meta" else np.array(json.dumps(v))) for k, v in arrays.items()
    }
    np.savez_compressed(path, **payload)
