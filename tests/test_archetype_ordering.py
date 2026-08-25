import itertools

import numpy as np
import pandas as pd

from archetypal_analysis.visualization.archetype_ordering import (
    adjacency_score,
    optimal_association_similarity_order,
    resolve_archetype_order,
)


def _result(similarity: pd.DataFrame) -> dict:
    return {
        "k": similarity.shape[1],
        "similarity_degree": similarity.to_numpy().tolist(),
        "index": similarity.index.tolist(),
        "columns": similarity.columns.tolist(),
    }


def test_top_association_groups_matching_categories_deterministically():
    similarity = pd.DataFrame(
        [
            [0.90, 0.75, 0.10, 0.05],
            [0.85, 0.70, 0.12, 0.08],
            [0.10, 0.20, 0.86, 0.76],
            [0.12, 0.18, 0.82, 0.72],
        ],
        index=["s1", "s2", "s3", "s4"],
        columns=[f"Archetype {i}" for i in range(1, 5)],
    )
    metadata = pd.DataFrame({"group": ["A", "A", "B", "B"]}, index=similarity.index)
    order = resolve_archetype_order(
        _result(similarity),
        metadata,
        {"target": "group", "legend_type": "discrete"},
        {"archetype_ordering": {"enabled": True, "method": "top_association"}},
    )
    assert order == ["Archetype 1", "Archetype 2", "Archetype 3", "Archetype 4"]


def test_association_similarity_exact_dp_matches_bruteforce_optimum():
    profiles = pd.DataFrame(
        [
            [1.00, 0.05, 0.00],
            [0.90, 0.15, 0.00],
            [0.15, 0.90, 0.10],
            [0.00, 0.20, 0.95],
            [0.00, 0.05, 1.00],
        ],
        index=[f"Archetype {i}" for i in range(1, 6)],
        columns=["A", "B", "C"],
    )
    chosen = optimal_association_similarity_order(profiles, metric="cosine", exact_max_k=15)
    chosen_score = adjacency_score(chosen, profiles, metric="cosine")

    brute_best = max(
        adjacency_score(order, profiles, metric="cosine")
        for order in itertools.permutations(profiles.index)
    )
    assert np.isclose(chosen_score, brute_best)


def test_explicit_order_overrides_metadata_ordering():
    similarity = pd.DataFrame(
        [[0.9, 0.1], [0.1, 0.9]],
        index=["s1", "s2"],
        columns=["Archetype 1", "Archetype 2"],
    )
    metadata = pd.DataFrame({"group": ["A", "B"]}, index=similarity.index)
    explicit = ["Archetype 2", "Archetype 1"]
    order = resolve_archetype_order(
        _result(similarity),
        metadata,
        {"target": "group", "legend_type": "discrete"},
        {
            "archetype_order": explicit,
            "archetype_ordering": {"enabled": True, "method": "association_similarity"},
        },
    )
    assert order == explicit


def test_large_k_fallback_is_deterministic_and_complete():
    rng = np.random.default_rng(7)
    profiles = pd.DataFrame(
        rng.random((18, 5)),
        index=[f"Archetype {i}" for i in range(1, 19)],
    )
    first = optimal_association_similarity_order(profiles, exact_max_k=10)
    second = optimal_association_similarity_order(profiles, exact_max_k=10)
    assert first == second
    assert set(first) == set(profiles.index)
