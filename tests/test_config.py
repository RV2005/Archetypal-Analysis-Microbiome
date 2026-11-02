import copy
from pathlib import Path
import os

<<<<<<< Updated upstream
import pytest

from archetypal_analysis.pipeline.config import ConfigError, load_config, validate_config
=======
from archetypal_analysis.pipeline.config import load_config, validate_config
from archetypal_analysis.cli import _configure_native_threading
>>>>>>> Stashed changes


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
    assert cfg["stability"]["heldout_projection"]["test_frac"] == 0.2

def test_visualization_ordering_config_defaults():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiments/compendium_sample_type.yaml")
    ordering = cfg["visualization"]["archetype_ordering"]
    assert cfg["visualization"]["reverse_order"] is False
    assert cfg["visualization"]["stability"]["cosine_ylim"] == [0.0, 1.0]
    assert cfg["visualization"]["stability"]["wasserstein_ylim"] == [0.0, 1.0]
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


<<<<<<< Updated upstream
def test_heldout_projection_test_frac_must_be_between_zero_and_one():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiments/dog_lifestyle.yaml")
    invalid = copy.deepcopy(cfg)
    invalid["stability"]["heldout_projection"]["test_frac"] = 1.0
    with pytest.raises(ConfigError, match="test_frac"):
        validate_config(invalid)
=======
def test_raw_quality_selection_modes_are_valid():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiments/dog_lifestyle.yaml")
    for mode in ("quality_silhouette", "quality_daviesbouldin"):
        candidate = dict(cfg)
        candidate["selection"] = dict(cfg["selection"], mode=mode)
        validate_config(candidate)


def test_native_threading_defaults_preserve_explicit_values(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "7")
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    _configure_native_threading()
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "7"
    assert os.environ["NUMBA_NUM_THREADS"] == "1"
    assert Path(os.environ["NUMBA_CACHE_DIR"]).is_dir()
>>>>>>> Stashed changes

def test_hadza_prediction_uses_grouped_cv_and_retains_subject_id():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiments/hadza_age.yaml")
    assert cfg["task"]["target"] == "age_months"
    assert cfg["prediction"]["group_column"] == "SUBJECT_ID"
    assert cfg["stability"]["sampling"]["group_column"] == "SUBJECT_ID"
