import matplotlib.pyplot as plt
import pandas as pd

from archetypal_analysis.visualization.stability import (
    plot_initialization_archetype_violin,
    plot_initialization_set_violin,
    plot_sampling_by_fraction_violin,
)


def test_cosine_and_wasserstein_stability_plots_default_to_zero_one_y_axis():
    cosine_frame = pd.DataFrame(
        {"reference_archetype": ["Archetype 1", "Archetype 2"], "max_cosine_similarity": [0.3, 0.8]}
    )
    wasserstein_frame = pd.DataFrame({"normalized_wasserstein_distance": [0.2, 0.7]})
    sampling_frame = pd.DataFrame({"fraction": [0.5, 0.8], "max_cosine_similarity": [0.4, 0.9]})

    figures = [
        plot_initialization_archetype_violin(cosine_frame, "max_cosine_similarity", {}),
        plot_initialization_set_violin(wasserstein_frame, "normalized_wasserstein_distance", {}),
        plot_sampling_by_fraction_violin(sampling_frame, "max_cosine_similarity", {}),
    ]
    try:
        assert all(figure.axes[0].get_ylim() == (0.0, 1.0) for figure in figures)
    finally:
        for figure in figures:
            plt.close(figure)
