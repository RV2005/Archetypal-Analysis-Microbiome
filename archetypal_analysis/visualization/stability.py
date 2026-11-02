"""Stability diagnostic plots."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _violin(ax, groups: Sequence[np.ndarray], labels: Sequence[str], ylabel: str, title: str):
    clean = [np.asarray(values, dtype=float)[np.isfinite(values)] for values in groups]
    positions = np.arange(1, len(clean) + 1)
    nonempty_positions = [p for p, values in zip(positions, clean) if len(values)]
    nonempty_groups = [values for values in clean if len(values)]
    if nonempty_groups:
        ax.violinplot(nonempty_groups, positions=nonempty_positions, showmedians=True, showextrema=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(labels), rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2, axis="y")
    return ax


def _apply_metric_ylim(ax, metric: str, cfg: Mapping[str, Any]) -> None:
    """Apply configured fixed limits for bounded stability metrics."""
    metric = metric.lower()
    if "cosine" in metric:
        ax.set_ylim(cfg.get("cosine_ylim", [0.0, 1.0]))
    elif "wasserstein" in metric:
        ax.set_ylim(cfg.get("wasserstein_ylim", [0.0, 1.0]))


def plot_initialization_archetype_violin(frame: pd.DataFrame, metric: str, cfg: Mapping[str, Any]):
    labels = sorted(frame["reference_archetype"].dropna().unique(), key=lambda x: int(str(x).split()[-1]))
    groups = [pd.to_numeric(frame.loc[frame["reference_archetype"] == label, metric], errors="coerce").to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=tuple(cfg.get("figsize", [10, 5])))
    _violin(ax, groups, labels, metric.replace("_", " ").title(), cfg.get("title", f"Initialization stability: {metric}"))
    _apply_metric_ylim(ax, metric, cfg)
    fig.tight_layout()
    return fig


def plot_initialization_set_violin(frame: pd.DataFrame, metric: str, cfg: Mapping[str, Any]):
    fig, ax = plt.subplots(figsize=tuple(cfg.get("figsize", [5, 5])))
    values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy()
    _violin(ax, [values], ["all runs"], metric.replace("_", " ").title(), cfg.get("title", f"Initialization stability: {metric}"))
    _apply_metric_ylim(ax, metric, cfg)
    fig.tight_layout()
    return fig


def plot_sampling_by_fraction_violin(frame: pd.DataFrame, metric: str, cfg: Mapping[str, Any]):
    fractions = sorted(pd.to_numeric(frame["fraction"], errors="coerce").dropna().unique())
    groups = [pd.to_numeric(frame.loc[np.isclose(pd.to_numeric(frame["fraction"], errors="coerce"), fraction), metric], errors="coerce").dropna().to_numpy() for fraction in fractions]
    labels = [f"{int(round(float(fraction) * 100))}%" for fraction in fractions]
    fig, ax = plt.subplots(figsize=tuple(cfg.get("figsize", [9, 5])))
    _violin(ax, groups, labels, metric.replace("_", " ").title(), cfg.get("title", f"Sampling stability: {metric}"))
    _apply_metric_ylim(ax, metric, cfg)
    ax.set_xlabel("Training sample fraction")
    fig.tight_layout()
    return fig
