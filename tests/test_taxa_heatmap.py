import numpy as np

from archetypal_analysis.visualization.pipeline_plots import plot_taxa_heatmap, taxa_heatmap_frame


def _result():
    return {
        "k": 2,
        "archetypes": [[8.0, 2.0, 0.0], [1.0, 3.0, 6.0]],
        "feature_names": ["Bacteria.A", "Bacteria.B", "Bacteria.C"],
    }


def _transform():
    return {"mode": "raw_counts", "standard_scale": False}


def test_taxa_heatmap_frame_is_percentage_composition():
    cfg = {"value_space": "fitted", "num_taxa": 3, "heatmap": {"num_taxa": 3}}
    frame = taxa_heatmap_frame(_result(), _transform(), cfg)
    assert frame.shape == (3, 2)
    assert np.allclose(frame.sum(axis=0).to_numpy(), [100.0, 100.0])
    assert frame.loc["Bacteria.A", "Archetype 1"] == 80.0


def test_taxa_heatmap_manual_and_auto_color_scales():
    base = {"value_space": "fitted", "num_taxa": 3, "simplified_taxa_level": -1}
    manual = {**base, "heatmap": {"num_taxa": 3, "scale": "manual", "vmin": 0, "vmax": 50}}
    fig = plot_taxa_heatmap(_result(), _transform(), manual)
    image = fig.axes[0].images[0]
    assert image.norm.vmin == 0
    assert image.norm.vmax == 50

    auto = {**base, "heatmap": {"num_taxa": 3, "scale": "auto"}}
    fig = plot_taxa_heatmap(_result(), _transform(), auto)
    image = fig.axes[0].images[0]
    assert image.norm.vmin == 0
    assert image.norm.vmax == 80
