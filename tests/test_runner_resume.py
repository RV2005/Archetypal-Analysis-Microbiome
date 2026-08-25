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
    assert first["qc"]["status"] == "ran"
    assert first["filter"]["status"] == "ran"
    assert reused == {"filter": reused["filter"]}
    assert reused["filter"]["status"] == "reused"
    assert runner.store.stage_marker("filter")["completed_at"] == before_filter

    overwritten = runner.run(["filter"], overwrite=True)
    assert overwritten["load"]["status"] == "ran"
    assert overwritten["qc"]["status"] == "ran"
    assert overwritten["filter"]["status"] == "ran"
