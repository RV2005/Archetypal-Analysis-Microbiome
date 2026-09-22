from pathlib import Path

from archetypal_analysis.pipeline.config import load_config


def test_experiment_inheritance():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiments/dog_lifestyle.yaml")
    assert cfg["experiment"]["name"] == "dog_lifestyle"
    assert cfg["dataset"]["otu_file"] == "otu.tsv"
    assert cfg["task"]["target"] == "Lifestyle"
    assert cfg["fit"]["k_values"] == [3, 4, 5, 6, 7, 8, 9, 10]
    assert cfg["selection"]["fixed_k"] == 8
    assert cfg["visualization"]["umap"]["n_neighbors"] == 15
    assert cfg["benchmarks"]["pca"]["enabled"] is True
    assert cfg["benchmarks"]["pcoa"]["distance_metric"] == "braycurtis"
    assert cfg["benchmarks"]["tsne"]["method"] == "auto"
    assert cfg["prediction"]["enabled"] is True
    assert cfg["prediction"]["logistic_regression"]["C_values"] == [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    assert cfg["prediction"]["logistic_regression"]["inner_cv_splits"] == 4
    assert cfg["stability"]["enabled"] is True

def test_visualization_ordering_config_defaults():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiments/compendium_sample_type.yaml")
    ordering = cfg["visualization"]["archetype_ordering"]
    assert ordering["method"] in {"top_association", "association_similarity"}
    assert ordering["similarity_metric"] in {"cosine", "euclidean"}
    assert ordering["exact_max_k"] >= 2
    assert cfg["visualization"]["k_summary"]["enabled"] is True
    assert cfg["analysis"]["taxa_value_space"] == "fitted"
    assert cfg["visualization"]["taxa"]["value_space"] == "fitted"
    assert cfg["visualization"]["taxa"]["heatmap"]["scale"] == "auto"
    assert cfg["analysis"]["metrics"]["convex_hull_max_dimension"] >= 2
    projection = cfg["visualization"]["archetype_projection"]
    assert projection["marker_mode"] in {"symbol", "numeric"}
    assert isinstance(projection["numeric_marker"], bool)
    assert projection["marker_legend"]["enabled"] is False

def test_hadza_prediction_uses_grouped_cv_and_retains_subject_id():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiments/hadza_age.yaml")
    assert cfg["task"]["target"] == "age_months"
    assert cfg["prediction"]["group_column"] == "SUBJECT_ID"
    assert cfg["stability"]["sampling"]["group_column"] == "SUBJECT_ID"
