"""Best-K scoring and selection policies."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    if finite.empty:
        return pd.Series(np.nan, index=s.index)
    lo, hi = finite.min(), finite.max()
    if np.isclose(hi, lo):
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out.loc[finite.index] = 1.0
        return out
    return (s - lo) / (hi - lo)


def add_quality_score(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add the requested normalized Silhouette - normalized Davies-Bouldin score."""
    out = metrics.copy()
    out["silhouette_normalized"] = minmax(out["silhouette"])
    # Higher raw DB is worse; keeping that direction and subtracting it exactly matches
    # "normalized silhouette - normalized Davies-Bouldin".
    out["davies_bouldin_normalized"] = minmax(out["davies_bouldin"])
    out["quality_score"] = out["silhouette_normalized"] - out["davies_bouldin_normalized"]
    return out


def _within_best_percent(values: pd.Series, best: float, percent: float, maximize: bool) -> pd.Series:
    """Robust relative tolerance around a best value, including zero/negative scores."""
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    spread = float(finite.max() - finite.min()) if not finite.empty else 0.0
    scale = abs(float(best))
    if scale < 1e-12:
        scale = max(spread, 1e-12)
    allowance = scale * (percent / 100.0)
    return values >= best - allowance if maximize else values <= best + allowance


def select_k(metrics: pd.DataFrame, cfg: Mapping[str, Any], k_values: list[int]) -> dict[str, Any]:
    mode = str(cfg.get("mode", "all")).lower()
    mode = {"none": "all", "skip": "all"}.get(mode, mode)

    if mode == "all":
        return {"mode": "all", "selected_k": sorted(k_values), "reason": "K selection skipped; downstream runs for every fitted K."}

    if mode == "fixed":
        k = int(cfg["fixed_k"])
        return {"mode": "fixed", "selected_k": [k], "reason": f"Hard-coded K={k}."}

    tolerance = float(cfg.get("within_percent", 5.0))
    frame = metrics.sort_values("k").copy()

    if mode == "quality":
        frame = add_quality_score(frame)
        valid = frame.dropna(subset=["quality_score"])
        if valid.empty:
            raise ValueError("No finite quality scores available for K selection")
        best = float(valid["quality_score"].max())
        eligible = valid.loc[_within_best_percent(valid["quality_score"], best, tolerance, maximize=True)]
        k = int(eligible["k"].min())
        return {
            "mode": "quality",
            "selected_k": [k],
            "within_percent": tolerance,
            "best_score": best,
            "reason": f"Lowest K with normalized silhouette - normalized Davies-Bouldin within {tolerance:g}% of the maximum.",
        }

    if mode == "reconstruction":
        metric = str(cfg.get("reconstruction_metric", "relative_frobenius_error"))
        if metric not in frame.columns:
            raise KeyError(f"Reconstruction metric '{metric}' not found in evaluation table")
        valid = frame.dropna(subset=[metric])
        if valid.empty:
            raise ValueError(f"No finite {metric} values available for K selection")
        best = float(valid[metric].min())
        eligible = valid.loc[_within_best_percent(valid[metric], best, tolerance, maximize=False)]
        k = int(eligible["k"].min())
        return {
            "mode": "reconstruction",
            "selected_k": [k],
            "within_percent": tolerance,
            "metric": metric,
            "best_error": best,
            "reason": f"Lowest K with {metric} within {tolerance:g}% of the lowest error.",
        }

    raise ValueError(f"Unsupported selection mode: {mode}")