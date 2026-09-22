import numpy as np

from archetypal_analysis.visualization.stacked_bar import stacked_bar_gradient_permutation


def test_stacked_bar_gradient_permutation_sorts_by_dominant_archetype_then_strength():
    points = np.array(
        [
            [0.1, 0.9, 0.0],
            [0.8, 0.2, 0.0],
            [0.6, 0.4, 0.0],
            [0.0, 0.7, 0.3],
        ]
    )

    assert stacked_bar_gradient_permutation(points) == [1, 2, 0, 3]


def test_stacked_bar_gradient_permutation_sorts_within_metadata_blocks():
    points = np.array(
        [
            [0.1, 0.9, 0.0],
            [0.8, 0.2, 0.0],
            [0.6, 0.4, 0.0],
            [0.0, 0.7, 0.3],
        ]
    )
    labels = np.array(["B", "A", "A", "B"])

    assert stacked_bar_gradient_permutation(points, labels=labels) == [1, 2, 0, 3]


def test_stacked_bar_gradient_permutation_keeps_missing_label_block():
    points = np.array(
        [
            [0.2, 0.8],
            [0.9, 0.1],
            [0.7, 0.3],
        ]
    )
    labels = np.array(["A", None, "A"], dtype=object)

    assert stacked_bar_gradient_permutation(points, labels=labels) == [2, 0, 1]