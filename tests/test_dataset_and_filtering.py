from pathlib import Path

import pandas as pd

from archetypal_analysis.data.dataset import DatasetBundle, UnifiedDatasetLoader
from archetypal_analysis.data.filtering import DatasetFilter


def test_unified_loader_aligns_samples(tmp_path: Path):
    otu = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["s1", "s2", "s3"])
    meta = pd.DataFrame({"group": ["x", "y", "z"]}, index=["s2", "s3", "s4"])
    otu.to_csv(tmp_path / "otu.tsv", sep="\t", index_label="sample")
    meta.to_csv(tmp_path / "metadata.tsv", sep="\t", index_label="sample")
    bundle = UnifiedDatasetLoader().load(tmp_path / "otu.tsv", tmp_path / "metadata.tsv", {"sample_id_column": "sample"})
    assert list(bundle.otu.index) == ["s2", "s3"]
    assert list(bundle.metadata.index) == ["s2", "s3"]


def test_filter_control_and_balance_are_deterministic():
    idx = [f"s{i}" for i in range(8)]
    otu = pd.DataFrame({"x": range(8)}, index=idx)
    meta = pd.DataFrame({
        "target": ["A"] * 4 + ["B"] * 4,
        "condition": ["healthy"] * 6 + ["other"] * 2,
    }, index=idx)
    cfg = {
        "drop_missing_target": True,
        "exclude_target_values": [],
        "control": {"condition": ["healthy"]},
        "normalize": {"target": None},
        "continuous": False,
        "ignore_zero": True,
        "ignore_values": [],
        "tolerance": 0.0,
        "random_state": 7,
    }
    out1 = DatasetFilter().apply(DatasetBundle(otu, meta), cfg, {"target": "target"})
    out2 = DatasetFilter().apply(DatasetBundle(otu, meta), cfg, {"target": "target"})
    assert list(out1.otu.index) == list(out2.otu.index)
    assert out1.metadata["target"].value_counts().to_dict() == {"A": 2, "B": 2}