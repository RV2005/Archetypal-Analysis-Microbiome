import json

import numpy as np
import pandas as pd

from archetypal_analysis.analysis.top_taxa_prediction import (
    evaluate_taxa_count,
    rank_archetype_taxa,
    taxa_counts,
)


def test_interleaved_ranking_deduplicates_and_count_grid_includes_full_input():
    archetypes = pd.DataFrame(
        [[9, 8, 1, 0], [0, 8, 9, 1]], columns=["a", "b", "c", "d"]
    )
    ranking = rank_archetype_taxa(archetypes, {"standard_scale": False})
    assert ranking == ["a", "c", "b", "d"]
    assert taxa_counts(73) == [10, 30, 50, 70, 73]


def test_top_taxa_settings_do_not_invalidate_main_prediction_stage():
    from archetypal_analysis.pipeline.stages.prediction import PredictionStage

    config = {
        "prediction": {"n_splits": 5, "top_taxa": {"random_forest": {"n_estimators": 10}}},
        "benchmarks": {}, "task": {}, "fit": {},
    }
    fragment = PredictionStage().config_fragment(config)
    assert fragment["prediction"] == {"n_splits": 5}


def test_forest_task_uses_fold_ranked_taxa_and_reports_out_of_fold_f1():
    samples = [f"s{i}" for i in range(20)]
    labels = pd.Series(["A"] * 10 + ["B"] * 10, index=samples)
    frame = pd.DataFrame({
        "signal": np.r_[np.zeros(10), np.ones(10)],
        "noise": np.arange(20, dtype=float),
    }, index=samples)
    folds = []
    for fold in range(2):
        test_ids = samples[fold::2]
        train_ids = [sample for sample in samples if sample not in test_ids]
        folds.append({
            "fold": fold, "train": frame.loc[train_ids], "test": frame.loc[test_ids],
            "ranking": ["signal", "noise"],
        })
    summary, predictions, fold_metrics, selected = evaluate_taxa_count(
        2, 1, folds, labels, "discrete", {"n_estimators": 20}, 17
    )
    assert summary["macro_f1_oof"] == 1.0
    assert len(predictions) == len(samples)
    assert len(fold_metrics) == 2
    assert set(selected["taxon"]) == {"signal"}


def test_forest_task_reports_r2_for_continuous_metadata():
    samples = [f"s{i}" for i in range(30)]
    signal = np.linspace(0, 10, 30)
    y = pd.Series(2 * signal + 1, index=samples)
    frame = pd.DataFrame({"signal": signal}, index=samples)
    folds = []
    for fold in range(3):
        test_ids = samples[fold::3]
        train_ids = [sample for sample in samples if sample not in test_ids]
        folds.append({
            "fold": fold, "train": frame.loc[train_ids], "test": frame.loc[test_ids],
            "ranking": ["signal"],
        })
    summary, _, _, _ = evaluate_taxa_count(
        2, 1, folds, y, "continuous", {"n_estimators": 50}, 11
    )
    assert summary["r2_oof"] > 0.9


def test_random_feature_baseline_is_reproducible_and_independent_of_k():
    samples = [f"s{i}" for i in range(24)]
    y = pd.Series(["A"] * 12 + ["B"] * 12, index=samples)
    frame = pd.DataFrame(
        np.arange(24 * 12, dtype=float).reshape(24, 12),
        index=samples, columns=[f"taxon_{i}" for i in range(12)],
    )
    folds = []
    for fold in range(2):
        test_ids = samples[fold::2]
        train_ids = [sample for sample in samples if sample not in test_ids]
        folds.append({
            "fold": fold, "train": frame.loc[train_ids], "test": frame.loc[test_ids],
            "ranking": frame.columns.tolist(),
        })
    args = (2, 5, folds, y, "discrete", {"n_estimators": 5}, 19)
    _, _, _, selected_0 = evaluate_taxa_count(*args, selection_mode="random", random_repeat=0)
    _, _, _, selected_again = evaluate_taxa_count(3, *args[1:], selection_mode="random", random_repeat=0)
    _, _, _, selected_1 = evaluate_taxa_count(*args, selection_mode="random", random_repeat=1)
    assert selected_0.equals(selected_again)
    assert not selected_0.equals(selected_1)
    assert selected_0.groupby("fold")["taxon"].nunique().tolist() == [5, 5]


def test_standalone_experiment_uses_existing_artifacts_without_pipeline_runner(tmp_path, monkeypatch):
    from archetypal_analysis.pipeline.artifacts import ArtifactStore
    from archetypal_analysis.pipeline.prediction_top_taxa import run_top_taxa_prediction

    config = {
        "experiment": {"name": "toy"},
        "paths": {"output_root": str(tmp_path)},
        "dataset": {"root": str(tmp_path), "otu_file": "unused.tsv", "metadata_file": "unused.tsv"},
        "task": {"target": "label", "legend_type": "discrete"},
        "fit": {"preprocessing": {"mode": "raw", "drop_zero_variance_features": True}},
        "prediction": {
            "n_splits": 2, "random_state": 3,
            "top_taxa": {
                "random_repeats": 3,
                "random_forest": {"n_estimators": 10},
                "parallel": {"backend": "process"},
            },
        },
        "analysis": {},
    }
    store = ArtifactStore(tmp_path / "toy")
    samples = [f"s{i}" for i in range(20)]
    labels = ["A"] * 10 + ["B"] * 10
    otu = pd.DataFrame({
        "signal": np.r_[np.linspace(1, 2, 10), np.linspace(4, 5, 10)],
        "noise": np.arange(20, dtype=float) + 1,
    }, index=samples)
    for i in range(9):
        otu[f"noise_{i}"] = np.sin(np.arange(20) + i) + i + 2
    otu.to_csv(store.stage_dir("fit") / "fit_input.tsv", sep="\t", index_label="sample")
    otu.to_csv(store.stage_dir("qc") / "otu.tsv", sep="\t", index_label="sample")
    pd.DataFrame({"label": labels}, index=samples).to_csv(
        store.stage_dir("qc") / "metadata.tsv", sep="\t", index_label="sample"
    )
    (store.stage_dir("select") / "selection.json").write_text(json.dumps({"selected_k": [2]}))

    def fake_aa(cache_dir, fingerprint, train, test, k, fit_cfg, projection_cfg):
        values = np.zeros((2, train.shape[1]))
        values[:, 0] = [2.0, 1.0]
        values[:, 1] = [1.0, 2.0]
        archetypes = pd.DataFrame(values, columns=train.columns, index=["Archetype 1", "Archetype 2"])
        return {"archetypes": archetypes}, {}, [], "ran"

    monkeypatch.setattr("archetypal_analysis.pipeline.prediction_top_taxa._load_cached_aa", fake_aa)
    out = run_top_taxa_prediction(config, n_jobs=2)
    scores = pd.read_csv(out / "scores.tsv", sep="\t")
    aa_scores = scores.loc[scores["method"] == "aa"].sort_values("num_taxa")
    random_scores = scores.loc[scores["method"] == "random"]
    assert aa_scores["num_taxa"].tolist() == [10, 11]
    assert len(random_scores) == 6
    assert aa_scores["macro_f1_oof"].iloc[0] > 0.5
    assert random_scores.loc[random_scores["num_taxa"] == 11, "macro_f1_oof"].eq(
        aa_scores.loc[aa_scores["num_taxa"] == 11, "macro_f1_oof"].iloc[0]
    ).all()
    assert (out / "score_vs_num_taxa.png").exists()
    assert (out / "K_002" / "N_0010" / "selected_taxa.tsv").exists()
    assert (out / "random" / "N_0010" / "repeat_000" / "selected_taxa.tsv").exists()
    assert json.loads((out / "manifest.json").read_text())["tasks_ran"] == 8
    run_top_taxa_prediction(config, n_jobs=2)
    assert json.loads((out / "manifest.json").read_text())["tasks_reused"] == 8
