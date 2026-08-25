import pandas as pd

from archetypal_analysis.analysis.selection import add_quality_score, select_k


def test_quality_selection_picks_lowest_k_within_tolerance():
    metrics = pd.DataFrame({
        "k": [3, 4, 5],
        "silhouette": [0.20, 0.398, 0.40],
        "davies_bouldin": [1.5, 1.005, 1.0],
        "relative_frobenius_error": [0.30, 0.25, 0.20],
    })
    scored = add_quality_score(metrics)
    result = select_k(scored, {"mode": "quality", "within_percent": 5.0}, [3, 4, 5])
    assert result["selected_k"] == [4]


def test_reconstruction_selection_picks_lowest_eligible_k():
    metrics = pd.DataFrame({
        "k": [3, 4, 5, 6],
        "silhouette": [0.1] * 4,
        "davies_bouldin": [1.0] * 4,
        "relative_frobenius_error": [0.30, 0.211, 0.205, 0.200],
    })
    result = select_k(
        metrics,
        {"mode": "reconstruction", "within_percent": 6.0, "reconstruction_metric": "relative_frobenius_error"},
        [3, 4, 5, 6],
    )
    assert result["selected_k"] == [4]


def test_all_and_fixed_modes():
    metrics = pd.DataFrame({"k": [3, 4], "silhouette": [0, 0], "davies_bouldin": [0, 0]})
    assert select_k(metrics, {"mode": "all"}, [3, 4])["selected_k"] == [3, 4]
    assert select_k(metrics, {"mode": "fixed", "fixed_k": 4}, [3, 4])["selected_k"] == [4]