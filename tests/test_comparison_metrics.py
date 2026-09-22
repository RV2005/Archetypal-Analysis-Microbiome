import numpy as np
import pandas as pd

from archetypal_analysis.analysis.comparison_metrics import (
    convex_hull_coverage_ratio,
    evaluate_representation,
    fuzzy_c_means,
    fuzzy_normalized_mutual_information,
    metadata_permanova_score,
)


def test_fuzzy_partition_and_metrics_are_finite_for_separated_groups():
    points = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.2], [5.0, 5.0], [5.1, 5.0], [5.0, 5.2]],
        index=[f"s{i}" for i in range(6)],
        columns=["D1", "D2"],
    )
    partition = fuzzy_c_means(points.to_numpy(), 2, random_state=1)
    metadata = pd.Series(["A", "A", "A", "B", "B", "B"], index=points.index)
    nmi, n = fuzzy_normalized_mutual_information(partition.memberships, metadata.reset_index(drop=True))
    assert n == 6
    assert nmi > 0.8

    original = points.rename(columns={"D1": "f1", "D2": "f2"})
    result = evaluate_representation(
        method="benchmark",
        dimension=2,
        representation=points,
        memberships=partition.memberships,
        original_input=original,
        metadata=metadata,
        task_cfg={"legend_type": "discrete"},
        evaluation_cfg={},
        metric_cfg={"convex_hull_max_dimension": 2},
    )
    assert result["silhouette"] > 0
    assert result["fuzzy_silhouette"] > 0
    assert result["metadata_permanova"] > 0
    assert 0 <= result["metadata_fuzzy_nmi"] <= 1
    assert result["reconstruction_rss"] >= 0


def test_continuous_permanova_uses_continuous_predictor_without_binning():
    index = [f"s{i}" for i in range(10)]
    age = pd.Series(np.arange(10, dtype=float), index=index)
    points = pd.DataFrame({"D1": age.to_numpy(), "D2": age.to_numpy() * 0.5}, index=index)
    score, mode, n, df1, df2 = metadata_permanova_score(points, age, "continuous")
    assert mode == "continuous_distance_based_pseudo_f"
    assert n == 10
    assert df1 == 1
    assert df2 == 8
    assert score > 0


def test_convex_hull_coverage_is_bounded_and_reports_dimension():
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
    ratio, used, rank = convex_hull_coverage_ratio(points, max_dimension=2)
    assert 0 <= ratio <= 1
    assert used == 2
    assert rank == 2
