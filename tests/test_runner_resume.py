from pathlib import Path

import pandas as pd

from archetypal_analysis.pipeline.artifacts import ArtifactStore
from archetypal_analysis.pipeline.runner import PipelineRunner


def test_overwrite_requested_filter_replays_transient_dependencies(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    otu = pd.DataFrame({"Bacteria.Firmicutes.Clostridia.GenusA": [6000, 7000]}, index=["s1", "s2"])
    meta = pd.DataFrame({"group": ["A", "B"]}, index=otu.index)
    otu.to_csv(data_dir / "otu.tsv", sep="\t", index_label="sample")
    meta.to_csv(data_dir / "metadata.tsv", sep="\t", index_label="sample")

    cfg = {
        "paths": {"project_root": str(tmp_path), "output_root": "results"},
        "experiment": {"name": "resume"},
        "dataset": {"root": "data", "otu_file": "otu.tsv", "metadata_file": "metadata.tsv", "sample_id_column": "sample"},
        "task": {"target": "group", "required_metadata_columns": [], "legend_type": "discrete"},
        "filter": {"enabled": True, "drop_missing_target": True, "exclude_target_values": [], "control": {}, "normalize": {}},
        "qc": {"enabled": False},
        "fit": {"k_values": [2]},
        "evaluation": {},
        "selection": {"mode": "all"},
        "analysis": {"modules": []},
        "visualization": {"enabled": False},
    }
    runner = PipelineRunner(cfg, verbose=False)
    first = runner.run(["filter"])
    before_filter = runner.store.stage_marker("filter")["completed_at"]
    reused = runner.run(["filter"])
    assert first["load"]["status"] == "ran"
    assert first["filter"]["status"] == "ran"
    assert reused == {"filter": reused["filter"]}
    assert reused["filter"]["status"] == "reused"
    assert runner.store.stage_marker("filter")["completed_at"] == before_filter

    overwritten = runner.run(["filter"], overwrite=True)
    assert overwritten["load"]["status"] == "ran"
    assert overwritten["filter"]["status"] == "ran"


def test_qc_runs_after_filter_and_persists_dataset_artifacts(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    otu = pd.DataFrame(
        {
            "Bacteria.Firmicutes.Clostridia.GenusA": [6000, None, 8000],
            "Bacteria.Firmicutes.NA.GenusB": [1, 0, 0],
            "Bacteria.Firmicutes.Clostridia.GenusC": [None, 7000, 1000],
        },
        index=["s1", "s2", "s3"],
    )
    meta = pd.DataFrame({"group": ["A", "B", None], "extra": [1, 2, 3]}, index=otu.index)
    otu.to_csv(data_dir / "otu.tsv", sep="\t", index_label="sample")
    meta.to_csv(data_dir / "metadata.tsv", sep="\t", index_label="sample")

    cfg = {
        "paths": {"project_root": str(tmp_path), "output_root": "results"},
        "experiment": {"name": "qc_after_filter"},
        "dataset": {"root": "data", "otu_file": "otu.tsv", "metadata_file": "metadata.tsv", "sample_id_column": "sample"},
        "task": {"target": "group", "required_metadata_columns": [], "legend_type": "discrete"},
        "filter": {"enabled": True, "drop_missing_target": True, "exclude_target_values": [], "control": {}, "normalize": {}},
        "qc": {"enabled": True, "min_taxa_level": 3, "min_nonzero_samples": 0, "min_counts": 0},
        "fit": {"k_values": [2]},
        "evaluation": {},
        "selection": {"mode": "all"},
        "analysis": {"modules": []},
        "visualization": {"enabled": False},
    }

    runner = PipelineRunner(cfg, verbose=False)
    result = runner.run(["qc"])
    qc_dir = runner.store.stage_dir("qc")

    assert list(result) == ["load", "filter", "qc"]
    assert (qc_dir / "otu.tsv").exists()
    assert (qc_dir / "metadata.tsv").exists()
    assert (qc_dir / "qc_summary.json").exists()
    assert pd.read_csv(qc_dir / "metadata.tsv", sep="\t", index_col="sample").index.tolist() == ["s1", "s2"]
    qc_otu = pd.read_csv(qc_dir / "otu.tsv", sep="\t", index_col="sample")
    assert qc_otu.shape == (2, 2)
    assert qc_otu.isna().sum().sum() == 0
    assert qc_otu.loc["s2", "Bacteria.Firmicutes.Clostridia.GenusA"] == 0

def test_runner_places_benchmark_before_analyze_and_analyze_depends_on_it():
    from archetypal_analysis.pipeline.runner import DEFAULT_STAGE_ORDER, DEFAULT_STAGE_SPECS

    assert DEFAULT_STAGE_ORDER.index("benchmark") < DEFAULT_STAGE_ORDER.index("analyze")
    assert DEFAULT_STAGE_SPECS["benchmark"][2] == ("fit",)
    assert DEFAULT_STAGE_SPECS["prediction"][2] == ("select",)
    assert "benchmark" in DEFAULT_STAGE_SPECS["analyze"][2]