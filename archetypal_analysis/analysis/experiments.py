"""Extensible registry for downstream analyses on fitted AA results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ..data.preprocessing import inverse_preprocessing

AnalysisFn = Callable[[dict[str, Any], pd.DataFrame, pd.DataFrame, Mapping[str, Any], Path], list[Path]]
ANALYSIS_REGISTRY: dict[str, AnalysisFn] = {}


def register_analysis(name: str):
    def decorator(func: AnalysisFn) -> AnalysisFn:
        ANALYSIS_REGISTRY[name] = func
        return func
    return decorator


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def similarity_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(result["similarity_degree"], index=pd.Index(result["index"], name="sample"), columns=result["columns"])


def inverse_archetypes(archetypes: np.ndarray, transform: Mapping[str, Any]) -> np.ndarray:
    """Map archetypes from fitted space back to an interpretable abundance scale.

    Raw/log modes return count-like values, relative abundance stays on its configured
    scale, and CLR is inverted to a closed composition whose rows sum to one.
    """
    arr = np.asarray(archetypes, dtype=float).copy()
    if transform.get("standard_scale"):
        mean = np.asarray(transform["mean"], dtype=float)
        std = np.asarray(transform["std"], dtype=float)
        arr = arr * std + mean
    arr = inverse_preprocessing(arr, transform)
    return np.clip(arr, a_min=0.0, a_max=None)


@register_analysis("sample_scores")
def sample_scores(result, metadata, fit_input, cfg, out_dir):
    sim = similarity_frame(result)
    target = cfg["task"]["target"]
    out = sim.copy()
    out[target] = metadata[target].reindex(out.index)
    out["dominant_archetype"] = sim.idxmax(axis=1)
    out["dominant_similarity"] = sim.max(axis=1)
    p = out_dir / "sample_scores.tsv"
    out.to_csv(p, sep="\t", index_label="sample")
    return [p]


@register_analysis("dominance")
def dominance(result, metadata, fit_input, cfg, out_dir):
    sim = similarity_frame(result)
    threshold = float(cfg["analysis"].get("dominance_threshold", 0.5))
    eps = 1e-12
    norm = sim.clip(lower=0)
    row_sums = norm.sum(axis=1).replace(0, np.nan)
    probs = norm.div(row_sums, axis=0).fillna(0)
    out = pd.DataFrame(index=sim.index)
    out["dominant_archetype"] = sim.idxmax(axis=1)
    out["max_similarity"] = sim.max(axis=1)
    out["is_dominant"] = out["max_similarity"] > threshold
    out["entropy"] = -(probs * np.log(probs + eps)).sum(axis=1)
    out["n_archetypes_above_threshold"] = (sim > threshold).sum(axis=1)
    p = out_dir / "dominance.tsv"
    out.to_csv(p, sep="\t", index_label="sample")
    return [p]


@register_analysis("metadata_association")
def metadata_association(result, metadata, fit_input, cfg, out_dir):
    sim = similarity_frame(result)
    target = cfg["task"]["target"]
    labels = metadata[target].reindex(sim.index)
    common = labels.notna()
    if cfg["task"].get("legend_type", "discrete") == "continuous":
        numeric = pd.to_numeric(labels[common], errors="coerce")
        out = sim.loc[common].corrwith(numeric).rename("pearson_r").to_frame()
        out.index.name = "archetype"
    else:
        out = sim.loc[common].assign(**{target: labels.loc[common]}).groupby(target).mean().T
        out.index.name = "archetype"
    p = out_dir / "metadata_association.tsv"
    out.to_csv(p, sep="\t")
    return [p]


@register_analysis("top_taxa")
def top_taxa(result, metadata, fit_input, cfg, out_dir):
    transform = json.loads((Path(cfg["_runtime"]["fit_dir"]) / "transform.json").read_text())
    arch = inverse_archetypes(np.asarray(result["archetypes"], dtype=float), transform)
    features = list(result["feature_names"])
    n = int(cfg["analysis"].get("num_top_taxa", cfg["visualization"].get("taxa", {}).get("num_taxa", 5)))
    rows = []
    for i, values in enumerate(arch):
        order = np.argsort(values)[::-1][:n]
        total = float(values.sum())
        for rank, j in enumerate(order, 1):
            rows.append({
                "archetype": f"Archetype {i + 1}",
                "rank": rank,
                "taxon": features[j],
                "value": float(values[j]),
                "fraction": float(values[j] / total) if total > 0 else np.nan,
            })
    out = pd.DataFrame(rows)
    p = out_dir / "top_taxa.tsv"
    out.to_csv(p, sep="\t", index=False)
    return [p]