from pathlib import Path

from archetypal_analysis.pipeline.config import load_config


def test_experiment_inheritance():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiments/dog_lifestyle.yaml")
    assert cfg["experiment"]["name"] == "dog_lifestyle"
    assert cfg["dataset"]["otu_file"] == "otu.tsv"
    assert cfg["task"]["target"] == "Lifestyle"
    assert cfg["fit"]["k_values"] == [5, 6, 7, 8]
    assert cfg["selection"]["fixed_k"] == 8
    assert cfg["visualization"]["umap"]["n_neighbors"] == 15

def test_visualization_ordering_config_defaults():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiments/compendium_sample_type.yaml")
    ordering = cfg["visualization"]["archetype_ordering"]
    assert ordering["method"] in {"top_association", "association_similarity"}
    assert ordering["similarity_metric"] in {"cosine", "euclidean"}
    assert ordering["exact_max_k"] >= 2
    assert cfg["visualization"]["k_summary"]["enabled"] is True
    projection = cfg["visualization"]["archetype_projection"]
    assert projection["marker_mode"] == "symbol"
    assert projection["numeric_marker"] is False
    assert projection["marker_legend"]["enabled"] is False
