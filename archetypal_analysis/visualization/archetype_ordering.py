"""Metadata-aware archetype ordering for simplex and archetype-indexed plots."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..analysis.experiments import similarity_frame


def natural_key(value: Any) -> list[Any]:
    """Natural-sort key so Archetype 2 sorts before Archetype 10."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", str(value))]


def clean_task_labels(metadata: pd.DataFrame, task: Mapping[str, Any]) -> pd.DataFrame:
    """Apply the same task label cleanup used by plotting before association scoring."""
    out = metadata.copy()
    target = task["target"]
    if target not in out.columns:
        raise KeyError(f"Target metadata column '{target}' not found")
    mask = out[target].notna()
    if task.get("strip_numeric_prefix", False):
        out.loc[mask, target] = (
            out.loc[mask, target]
            .astype(str)
            .str.replace(r"^\s*\d+\s*[\.\):-]?\s*", "", regex=True)
            .str.strip()
        )
    rename = task.get("label_rename_map", {}) or {}
    if rename:
        out.loc[mask, target] = out.loc[mask, target].map(lambda x: rename.get(x, x))
    return out


def metadata_association_profiles(
    result: Mapping[str, Any],
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
) -> pd.DataFrame:
    """Return one metadata-association vector per archetype.

    For categorical targets, component ``c`` is the mean similarity degree of an
    archetype among samples in category ``c``. This is the same quantity written by
    the built-in ``metadata_association`` downstream analysis.

    For continuous targets, each archetype has one Pearson correlation coefficient.
    """
    sim = similarity_frame(result)
    target = task["target"]
    labels = clean_task_labels(metadata, task)[target].reindex(sim.index)
    valid = labels.notna()
    sim = sim.loc[valid]
    labels = labels.loc[valid]
    if sim.empty:
        return pd.DataFrame(index=sim.columns)

    if task.get("legend_type", "discrete") == "continuous":
        numeric = pd.to_numeric(labels, errors="coerce")
        profile = sim.corrwith(numeric).rename("pearson_r").to_frame().fillna(0.0)
        profile.index.name = "archetype"
        return profile

    profile = sim.assign(**{target: labels}).groupby(target).mean().T.fillna(0.0)
    # Deterministic category ordering makes ties/repeated runs reproducible.
    profile = profile.reindex(sorted(profile.columns, key=natural_key), axis=1)
    profile.index.name = "archetype"
    return profile


def top_association_order(profiles: pd.DataFrame) -> list[str]:
    """Group archetypes by their strongest metadata category deterministically."""
    if profiles.empty:
        return list(profiles.index)

    # A continuous target only has a single correlation component. Put archetypes
    # along that one-dimensional association axis.
    if profiles.shape[1] == 1:
        values = profiles.iloc[:, 0]
        return sorted(values.index, key=lambda name: (float(values.loc[name]), natural_key(name)))

    grouped: dict[Any, list[tuple[str, float]]] = {}
    for name, row in profiles.iterrows():
        maximum = float(row.max())
        tied = [category for category, value in row.items() if np.isclose(float(value), maximum)]
        top_category = sorted(tied, key=natural_key)[0]
        grouped.setdefault(top_category, []).append((str(name), maximum))

    ordered: list[str] = []
    for category in sorted(grouped, key=natural_key):
        # Within one category, stronger association first, then natural archetype name.
        members = sorted(grouped[category], key=lambda item: (-item[1], natural_key(item[0])))
        ordered.extend(name for name, _ in members)
    return ordered


def _pairwise_profile_similarity(profiles: pd.DataFrame, metric: str) -> np.ndarray:
    values = np.asarray(profiles, dtype=float)
    n = len(values)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    # With a continuous target there is only one association coefficient per
    # archetype. Closeness on that axis is more meaningful than cosine similarity.
    if values.shape[1] == 1:
        delta = np.abs(values[:, 0, None] - values[None, :, 0])
        scale = float(np.nanmax(delta)) if delta.size else 0.0
        similarity = 1.0 - (delta / scale if scale > 0 else delta)
        np.fill_diagonal(similarity, 0.0)
        return similarity

    metric = str(metric).lower()
    if metric == "cosine":
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
        similarity = normalized @ normalized.T
    elif metric in {"euclidean", "negative_euclidean"}:
        delta = values[:, None, :] - values[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        scale = float(np.nanmax(distance)) if distance.size else 0.0
        similarity = 1.0 - (distance / scale if scale > 0 else distance)
    else:
        raise ValueError("archetype_ordering.similarity_metric must be 'cosine' or 'euclidean'")

    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def adjacency_score(order: Sequence[str], profiles: pd.DataFrame, metric: str = "cosine") -> float:
    """Score an ordering by the sum of metadata-profile similarity of neighbors."""
    if len(order) < 2:
        return 0.0
    aligned = profiles.loc[list(order)]
    similarity = _pairwise_profile_similarity(aligned, metric)
    return float(sum(similarity[i, i + 1] for i in range(len(order) - 1)))


def _canonical_direction(order: list[str]) -> list[str]:
    """An open path and its reverse are equivalent; choose one deterministically."""
    reverse = list(reversed(order))
    key_forward = tuple(tuple(natural_key(name)) for name in order)
    key_reverse = tuple(tuple(natural_key(name)) for name in reverse)
    return order if key_forward <= key_reverse else reverse


def _held_karp_max_path(names: Sequence[str], weights: np.ndarray) -> list[str]:
    """Exact maximum-weight Hamiltonian path in O(K^2 2^K) time/O(K 2^K) memory."""
    n = len(names)
    if n <= 2:
        return _canonical_direction(list(names))

    size = 1 << n
    neg_inf = -np.inf
    scores = np.full((size, n), neg_inf, dtype=float)
    parents = np.full((size, n), -1, dtype=np.int16 if n < 32767 else np.int32)
    for i in range(n):
        scores[1 << i, i] = 0.0

    for mask in range(1, size):
        for end in range(n):
            bit = 1 << end
            if not mask & bit:
                continue
            previous_mask = mask ^ bit
            if previous_mask == 0:
                continue
            best_score = neg_inf
            best_previous = -1
            for previous in range(n):
                if not previous_mask & (1 << previous):
                    continue
                candidate = scores[previous_mask, previous] + float(weights[previous, end])
                if candidate > best_score + 1e-12 or (
                    abs(candidate - best_score) <= 1e-12 and (best_previous < 0 or previous < best_previous)
                ):
                    best_score = candidate
                    best_previous = previous
            scores[mask, end] = best_score
            parents[mask, end] = best_previous

    full = size - 1
    best_value = float(np.max(scores[full]))
    end_candidates = np.flatnonzero(np.isclose(scores[full], best_value, rtol=0.0, atol=1e-12))
    end = int(end_candidates[0])
    indices: list[int] = []
    mask = full
    while end >= 0:
        indices.append(end)
        previous = int(parents[mask, end])
        mask ^= 1 << end
        end = previous
    indices.reverse()
    return _canonical_direction([str(names[i]) for i in indices])


def _greedy_path(start: int, weights: np.ndarray) -> list[int]:
    n = len(weights)
    remaining = set(range(n))
    remaining.remove(start)
    path = [start]
    while remaining:
        current = path[-1]
        nxt = min(remaining, key=lambda j: (-float(weights[current, j]), j))
        path.append(nxt)
        remaining.remove(nxt)
    return path


def _path_score_indices(path: Sequence[int], weights: np.ndarray) -> float:
    return float(sum(weights[path[i], path[i + 1]] for i in range(len(path) - 1)))


def _two_opt_open_path(path: list[int], weights: np.ndarray) -> list[int]:
    """Deterministic 2-opt local improvement for large-K fallback paths."""
    best = list(path)
    improved = True
    while improved:
        improved = False
        best_score = _path_score_indices(best, weights)
        for i in range(len(best) - 2):
            for j in range(i + 2, len(best)):
                candidate = best[: i + 1] + list(reversed(best[i + 1 : j + 1])) + best[j + 1 :]
                score = _path_score_indices(candidate, weights)
                if score > best_score + 1e-12:
                    best = candidate
                    best_score = score
                    improved = True
                    break
            if improved:
                break
    return best


def optimal_association_similarity_order(
    profiles: pd.DataFrame,
    *,
    metric: str = "cosine",
    exact_max_k: int = 15,
) -> list[str]:
    """Maximize the sum of metadata-association similarity of adjacent archetypes.

    This is an open maximum-weight Hamiltonian-path problem. For typical AA values of
    K, Held-Karp dynamic programming solves it exactly. Above ``exact_max_k`` a
    deterministic multi-start greedy path followed by 2-opt is used to avoid the
    exponential memory/time cost.
    """
    names = [str(name) for name in profiles.index]
    n = len(names)
    if n <= 2:
        return _canonical_direction(names)
    weights = _pairwise_profile_similarity(profiles, metric)

    if n <= int(exact_max_k):
        return _held_karp_max_path(names, weights)

    candidates = [_two_opt_open_path(_greedy_path(start, weights), weights) for start in range(n)]
    best = min(
        candidates,
        key=lambda path: (-_path_score_indices(path, weights), tuple(path)),
    )
    return _canonical_direction([names[i] for i in best])


def resolve_archetype_order(
    result: Mapping[str, Any],
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> list[str]:
    """Resolve explicit or metadata-derived archetype ordering."""
    names = list(similarity_frame(result).columns)
    explicit = cfg.get("archetype_order")
    if explicit and len(explicit) == len(names) and set(explicit) == set(names):
        return list(explicit)

    ordering_cfg = cfg.get("archetype_ordering", {}) or {}
    if not ordering_cfg.get("enabled", False):
        return names

    profiles = metadata_association_profiles(result, metadata, task).reindex(names)
    if profiles.empty:
        return names

    method = str(ordering_cfg.get("method", "association_similarity")).lower()
    if method == "top_association":
        return top_association_order(profiles)
    if method == "association_similarity":
        return optimal_association_similarity_order(
            profiles,
            metric=str(ordering_cfg.get("similarity_metric", "cosine")),
            exact_max_k=int(ordering_cfg.get("exact_max_k", 15)),
        )
    raise ValueError(
        "visualization.archetype_ordering.method must be 'top_association' or 'association_similarity'"
    )