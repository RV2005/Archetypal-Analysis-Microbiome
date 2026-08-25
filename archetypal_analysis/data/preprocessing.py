"""Configurable preprocessing transforms for archetypal-analysis inputs.

The pipeline performs filtering and QC on the original count table, then applies exactly
one preprocessing mode before fitting.  Keeping this logic in one module makes the
transform reusable from notebooks/tests and records enough metadata for downstream
interpretation of fitted archetypes.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


PREPROCESSING_MODES = ("raw_counts", "log", "relative_abundance", "clr")
_MODE_ALIASES = {
    "raw": "raw_counts",
    "counts": "raw_counts",
    "raw_count": "raw_counts",
    "log1p": "log",
    "log_transform": "log",
    "relative": "relative_abundance",
    "rel_abundance": "relative_abundance",
    "relative-abundance": "relative_abundance",
    "centered_log_ratio": "clr",
    "centered-log-ratio": "clr",
}


def resolve_preprocessing_mode(cfg: Mapping[str, Any] | None) -> str:
    """Resolve canonical preprocessing mode, including the legacy log flag.

    Before the mode option existed, ``log_transform`` controlled the fit transform.
    Preserve that behavior for old experiment YAMLs; if neither key is provided the
    historical default is the log2(x + 1) transform.
    """
    cfg = dict(cfg or {})
    if cfg.get("mode") is None:
        if "log_transform" in cfg:
            mode = "log" if bool(cfg["log_transform"]) else "raw_counts"
        else:
            mode = "log"
    else:
        mode = str(cfg["mode"]).strip().lower()
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in PREPROCESSING_MODES:
        raise ValueError(
            f"Unknown preprocessing mode '{mode}'. Expected one of {list(PREPROCESSING_MODES)}"
        )
    return mode


def _validate_nonnegative(data: pd.DataFrame, mode: str) -> None:
    values = data.to_numpy(dtype=float, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size and np.min(finite) < 0:
        raise ValueError(f"Preprocessing mode '{mode}' requires non-negative input values")


def apply_preprocessing(
    otu: pd.DataFrame,
    cfg: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one configured transform and return data plus transform metadata.

    Modes
    -----
    raw_counts
        No abundance transform.
    log
        Current AA behavior: ``log_base(x + pseudocount)``. Defaults to log2(x + 1).
    relative_abundance
        Divide each sample by its row sum and multiply by ``relative_abundance_scale``.
    clr
        Add ``clr_pseudocount``, take natural logs, and subtract each sample's mean log
        abundance. CLR is scale invariant; inverse CLR therefore returns compositions.
    """
    cfg = dict(cfg or {})
    mode = resolve_preprocessing_mode(cfg)
    data = otu.copy().astype(float)

    metadata: dict[str, Any] = {
        "mode": mode,
        # Retain the legacy field so older downstream code/artifacts remain readable.
        "log_transform": mode == "log",
        "log_base": float(cfg.get("log_base", 2.0)),
        "pseudocount": float(cfg.get("pseudocount", 1.0)),
        "relative_abundance_scale": float(cfg.get("relative_abundance_scale", 1.0)),
        "clr_pseudocount": float(cfg.get("clr_pseudocount", 1.0)),
    }

    _validate_nonnegative(data, mode)

    if mode == "raw_counts":
        return data, metadata

    if mode == "log":
        base = metadata["log_base"]
        pseudocount = metadata["pseudocount"]
        if base <= 0 or math.isclose(base, 1.0):
            raise ValueError("fit.preprocessing.log_base must be positive and != 1")
        if pseudocount <= 0:
            raise ValueError("fit.preprocessing.pseudocount must be > 0 for log mode")
        data = np.log(data + pseudocount) / np.log(base)
        return data, metadata

    if mode == "relative_abundance":
        scale = metadata["relative_abundance_scale"]
        if scale <= 0:
            raise ValueError("fit.preprocessing.relative_abundance_scale must be > 0")
        row_sums = data.sum(axis=1)
        data = data.div(row_sums.replace(0, np.nan), axis=0) * scale
        return data, metadata

    # CLR uses the natural logarithm by definition. A positive pseudocount handles zeros.
    pseudocount = metadata["clr_pseudocount"]
    if pseudocount <= 0:
        raise ValueError("fit.preprocessing.clr_pseudocount must be > 0")
    logged = np.log(data + pseudocount)
    data = logged.sub(logged.mean(axis=1), axis=0)
    return data, metadata


def inverse_preprocessing(values: np.ndarray, transform: Mapping[str, Any]) -> np.ndarray:
    """Map fitted-space values back to an interpretable abundance representation.

    For CLR the original library size is intentionally not recoverable, so the inverse
    is the closure of ``exp(clr)``: rows sum to 1 and can be interpreted as compositions.
    """
    arr = np.asarray(values, dtype=float).copy()
    mode = transform.get("mode")
    if mode is None:
        # Backward compatibility with transform.json files written before modes existed.
        mode = "log" if transform.get("log_transform") else "raw_counts"
    mode = _MODE_ALIASES.get(str(mode).lower(), str(mode).lower())

    if mode == "log":
        base = float(transform.get("log_base", 2.0))
        pseudo = float(transform.get("pseudocount", 1.0))
        return np.power(base, arr) - pseudo

    if mode == "clr":
        # Subtracting the row max is numerically stable and does not change closure.
        shifted = arr - np.max(arr, axis=1, keepdims=True)
        composition = np.exp(shifted)
        denom = composition.sum(axis=1, keepdims=True)
        return np.divide(composition, denom, out=np.zeros_like(composition), where=denom != 0)

    # Raw counts and relative abundance are already directly interpretable.  Relative
    # abundance may be configured to sum to a scale other than one; preserve that scale.
    return arr