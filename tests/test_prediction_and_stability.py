import json

import numpy as np
import pandas as pd

from archetypal_analysis.analysis.benchmarks import TrainTestBenchmark, fit_benchmark_train_test
from archetypal_analysis.analysis.prediction import (
    build_cv_plan,
    cross_validated_prediction,
    helmert_basis,
    helmert_transform_memberships,
)
from archetypal_analysis.analysis.prediction_representations import AAFoldRepresentation
from archetypal_analysis.analysis.stability import (
    compare_archetypes,
    normalized_wasserstein_distance,
    project_onto_archetypes,
    stratified_subsample_ids,
)


def test_grouped_cv_never_splits_an_individual_across_folds():
    groups = [f"p{i}" for i in range(6) for _ in range(2)]
    labels = ["A"] * 6 + ["B"] * 6
    index = [f"s{i}" for i in range(12)]
    metadata = pd.DataFrame({"target": labels, "person": groups}, index=index)
    plan, y, group_series = build_cv_plan(
        metadata, "target", "discrete", n_splits=3, random_state=4, group_column="person"
    )
    folds = plan.frame(group_series)
    assert folds.groupby("group")["fold"].nunique().max() == 1
    assert plan.grouped is True
    assert len(y) == 12


def test_helmert_basis_is_orthonormal_and_removes_sum_direction():
    h = helmert_basis(5)
    assert h.shape == (5, 4)
    assert np.allclose(h.T @ h, np.eye(4))
    assert np.allclose(np.ones(5) @ h, np.zeros(4))

    memberships = pd.DataFrame(
        [[0.5, 0.2, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.2, 0.2]],
        index=["a", "b"],
    )
    coords = helmert_transform_memberships(memberships)
    assert isinstance(coords, pd.DataFrame)
    assert coords.shape == (2, 4)
    # Pairwise Euclidean distance is preserved because membership differences already
    # lie in the sum-zero subspace.
    assert np.isclose(
        np.linalg.norm(memberships.iloc[0].to_numpy() - memberships.iloc[1].to_numpy()),
        np.linalg.norm(coords.iloc[0].to_numpy() - coords.iloc[1].to_numpy()),
    )


def test_cross_validated_categorical_prediction_selects_C_from_grid():
    index = [f"s{i}" for i in range(40)]
    labels = pd.Series(["A"] * 20 + ["B"] * 20, index=index)
    metadata = pd.DataFrame({"target": labels}, index=index)
    representation = pd.DataFrame(
        {"D1": np.r_[np.linspace(-2, -1, 20), np.linspace(1, 2, 20)], "D2": np.zeros(40)},
        index=index,
    )
    plan, y, _ = build_cv_plan(metadata, "target", "discrete", n_splits=4, random_state=1)
    predictions, folds, summary = cross_validated_prediction(
        representation,
        y,
        plan,
        {
            "random_state": 1,
            "logistic_regression": {
                "C_values": [0.01, 0.1, 1.0, 10.0],
                "inner_cv_splits": 3,
                "max_iter": 1000,
            },
        },
        legend_type="discrete",
    )
    assert len(predictions) == 40
    assert len(folds) == 4
    assert set(folds["selected_C"]).issubset({0.01, 0.1, 1.0, 10.0})
    assert (folds["inner_cv_splits"] >= 2).all()
    assert summary["balanced_accuracy_mean"] > 0.9


def test_cross_validated_continuous_prediction_reports_r2_and_mae():
    index = [f"s{i}" for i in range(30)]
    x = np.linspace(0, 10, 30)
    metadata = pd.DataFrame({"age": 2 * x + 1}, index=index)
    representation = pd.DataFrame({"D1": x, "D2": x ** 2 * 0.0}, index=index)
    plan, y, _ = build_cv_plan(metadata, "age", "continuous", n_splits=5, random_state=2)
    _pred, _folds, summary = cross_validated_prediction(
        representation, y, plan, {"random_state": 2}, legend_type="continuous"
    )
    assert summary["r2_mean"] > 0.99
    assert summary["mae_mean"] < 1e-8


def test_pca_fold_benchmark_fits_train_only_and_transforms_test():
    train = pd.DataFrame(
        np.arange(30, dtype=float).reshape(10, 3), index=[f"tr{i}" for i in range(10)], columns=list("abc")
    )
    test = pd.DataFrame(
        np.arange(12, dtype=float).reshape(4, 3), index=[f"te{i}" for i in range(4)], columns=list("abc")
    )
    fitted = fit_benchmark_train_test(train, test, "pca", 2, {})
    assert fitted.train.shape == (10, 2)
    assert fitted.test is not None and fitted.test.shape == (4, 2)
    assert fitted.metadata["out_of_sample_method"] == "pca_transform"



def test_pcoa_fold_projection_preserves_euclidean_train_test_distances():
    from scipy.spatial import distance

    train = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        index=["a", "b", "c", "d"],
        columns=["x", "y"],
    )
    test = pd.DataFrame(
        [[0.2, 0.3], [2.0, 1.0]], index=["e", "f"], columns=["x", "y"]
    )
    fitted = fit_benchmark_train_test(train, test, "pcoa", 2, {"distance_metric": "euclidean"})
    embedded = distance.cdist(fitted.test.to_numpy(), fitted.train.to_numpy())
    original = distance.cdist(test.to_numpy(), train.to_numpy())
    assert np.allclose(embedded, original, atol=1e-10)
    assert fitted.metadata["out_of_sample_method"] == "gower_supplementary_projection"

def test_tsne_strict_oos_is_reported_unavailable_instead_of_leaking_test_data():
    train = pd.DataFrame(np.random.RandomState(1).normal(size=(12, 4)))
    test = pd.DataFrame(np.random.RandomState(2).normal(size=(3, 4)), columns=train.columns)
    fitted = fit_benchmark_train_test(train, test, "tsne", 2, {})
    assert fitted.test is None
    assert fitted.metadata["available"] is False
    assert "no transform" in fitted.metadata["reason"]


def test_archetype_matching_is_permutation_invariant_and_wasserstein_zero():
    reference = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    candidate = reference[[2, 0, 1]]
    table, overall = compare_archetypes(reference, candidate, matching_metric="cosine")
    assert np.allclose(table["matched_cosine_similarity"], 1.0)
    assert np.allclose(table["min_euclidean_distance"], 0.0)
    assert overall["normalized_wasserstein_distance"] == 0.0
    assert normalized_wasserstein_distance(reference, candidate) == 0.0


def test_heldout_projection_recovers_convex_combinations():
    archetypes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    expected = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    samples = expected @ archetypes
    projected, info = project_onto_archetypes(samples, archetypes, max_iter=5000, tol=1e-10)
    assert np.allclose(projected.sum(axis=1), 1.0)
    assert np.all(projected >= -1e-12)
    assert np.allclose(projected @ archetypes, samples, atol=1e-5)
    assert info["rss"] < 1e-8


def test_grouped_sampling_keeps_groups_entirely_train_or_heldout():
    index = [f"s{i}" for i in range(20)]
    metadata = pd.DataFrame(
        {
            "target": ["A"] * 10 + ["B"] * 10,
            "person": [f"p{i}" for i in range(10) for _ in range(2)],
        },
        index=index,
    )
    train, heldout = stratified_subsample_ids(
        metadata, "target", "discrete", 0.8, 123, group_column="person"
    )
    assert set(metadata.loc[train, "person"]).isdisjoint(set(metadata.loc[heldout, "person"]))


def test_group_columns_do_not_invalidate_filtered_aa_metadata():
    from archetypal_analysis.pipeline.stages.filter import required_metadata_columns

    cfg = {
        "task": {"target": "age_months", "required_metadata_columns": ["location"]},
        "prediction": {"group_column": "SUBJECT_ID"},
        "stability": {"sampling": {"group_column": None}},
    }
    assert required_metadata_columns(cfg) == ["location", "age_months"]


def _fake_aa_fit(train, test, k, fit_cfg, projection_cfg):
    # Deterministic soft memberships derived only from each passed fold matrix.
    def memberships(frame):
        signal = frame.iloc[:, 0].to_numpy(dtype=float)
        base = np.column_stack([np.exp(-(signal - center) ** 2) for center in np.linspace(signal.min(), signal.max(), k)])
        base /= base.sum(axis=1, keepdims=True)
        return pd.DataFrame(base, index=frame.index, columns=[f"Archetype {i+1}" for i in range(k)])

    tr = memberships(train)
    te = memberships(test)
    from archetypal_analysis.analysis.prediction import helmert_transform_memberships

    return AAFoldRepresentation(
        train_direct=tr,
        test_direct=te,
        train_helmert=helmert_transform_memberships(tr),
        test_helmert=helmert_transform_memberships(te),
        archetypes=pd.DataFrame(
            np.zeros((k, train.shape[1])),
            index=tr.columns,
            columns=train.columns,
        ),
        metadata={"k": k, "intrinsic_dimension": k - 1},
    )


def _fake_benchmark_fit(train, test, method, dimension, benchmark_cfg):
    columns = [f"D{i+1}" for i in range(dimension)]
    train_values = np.column_stack(
        [train.iloc[:, i % train.shape[1]].to_numpy(dtype=float) for i in range(dimension)]
    )
    test_values = np.column_stack(
        [test.iloc[:, i % test.shape[1]].to_numpy(dtype=float) for i in range(dimension)]
    )
    return TrainTestBenchmark(
        pd.DataFrame(train_values, index=train.index, columns=columns),
        pd.DataFrame(test_values, index=test.index, columns=columns),
        {"method": method, "dimension": dimension, "out_of_sample_method": "test_fake"},
    )


def test_prediction_stage_refits_representations_per_fold_and_uses_k_minus_one(tmp_path, monkeypatch):
    from archetypal_analysis.pipeline.artifacts import ArtifactStore
    from archetypal_analysis.pipeline.stages.base import StageContext
    import archetypal_analysis.pipeline.stages.prediction as stage_module

    store = ArtifactStore(tmp_path / "results")
    samples = [f"s{i}" for i in range(30)]
    y = np.array(["A"] * 15 + ["B"] * 15)
    signal = np.r_[np.linspace(1, 2, 15), np.linspace(4, 5, 15)]

    qc_dir = store.stage_dir("qc")
    pd.DataFrame({"f1": signal, "f2": signal * 0.5, "f3": np.arange(30)}, index=samples).to_csv(
        qc_dir / "otu.tsv", sep="\t", index_label="sample"
    )
    pd.DataFrame({"target": y}, index=samples).to_csv(
        qc_dir / "metadata.tsv", sep="\t", index_label="sample"
    )
    fit_dir = store.stage_dir("fit")
    pd.DataFrame({"f1": signal, "f2": signal * 0.5, "f3": np.arange(30)}, index=samples).to_csv(
        fit_dir / "fit_input.tsv", sep="\t", index_label="sample"
    )
    select_dir = store.stage_dir("select")
    (select_dir / "selection.json").write_text(json.dumps({"selected_k": [3]}))

    aa_calls = []
    benchmark_calls = []

    def fake_aa(*args, **kwargs):
        aa_calls.append((tuple(args[0].index), tuple(args[1].index), args[2]))
        return _fake_aa_fit(*args, **kwargs)

    def fake_benchmark(*args, **kwargs):
        benchmark_calls.append((args[2], args[3], tuple(args[0].index), tuple(args[1].index)))
        return _fake_benchmark_fit(*args, **kwargs)

    monkeypatch.setattr(stage_module, "fit_aa_train_test", fake_aa)
    monkeypatch.setattr(stage_module, "fit_benchmark_fold", fake_benchmark)

    cfg = {
        "prediction": {
            "enabled": True,
            "n_splits": 3,
            "random_state": 3,
            "raw_baseline": "qc_otu",
            "logistic_regression": {"C_values": [0.1, 1.0], "inner_cv_splits": 2, "max_iter": 1000},
        },
        "benchmarks": {
            "pca": {"enabled": True},
            "pcoa": {"enabled": False},
            "umap": {"enabled": False},
            "tsne": {"enabled": False},
        },
        "task": {"target": "target", "legend_type": "discrete"},
        "fit": {"k_values": [3], "preprocessing": {"mode": "raw_counts"}, "random_state": 7},
    }
    result = stage_module.PredictionStage().run(StageContext(config=cfg, store=store, verbose=False))
    scores = pd.read_csv(store.stage_dir("prediction") / "prediction_scores.tsv", sep="\t")

    assert result["status"] == "ran"
    assert len(aa_calls) == 3  # once per outer fold, never the global all-data AA artifact
    assert len(benchmark_calls) == 3
    assert all(call[1] == 2 for call in benchmark_calls)  # K=3 -> K-1=2
    assert set(scores["method"]) == {
        "raw_input",
        "archetypal_analysis_direct",
        "archetypal_analysis_helmert",
        "pca",
    }
    direct = scores.loc[scores["method"] == "archetypal_analysis_direct"].iloc[0]
    helmert = scores.loc[scores["method"] == "archetypal_analysis_helmert"].iloc[0]
    pca = scores.loc[scores["method"] == "pca"].iloc[0]
    assert direct["representation_dimension"] == 3
    assert direct["effective_dimension"] == 2
    assert helmert["representation_dimension"] == 2
    assert pca["representation_dimension"] == 2


def test_group_resolver_fills_null_preferred_subject_id_from_canonical_metadata(tmp_path):
    from archetypal_analysis.data.grouping import resolve_group_labels

    samples = [f"s{i}" for i in range(8)]
    retained = pd.DataFrame({"age_months": np.arange(8), "SUBJECT_ID": [np.nan] * 8}, index=samples)
    source = pd.DataFrame(
        {
            "sample": samples,
            "SUBJECT_ID": [np.nan] * 8,
            "SUBJECT_ID2": ["c1", "c1", "c2", "c2", np.nan, np.nan, np.nan, np.nan],
            "SubjectID": [np.nan, np.nan, np.nan, np.nan, "c3", "c3", "c4", "c4"],
            "location": ["Finland"] * 4 + ["Germany"] * 4,
        }
    )
    path = tmp_path / "metadata.tsv"
    source.to_csv(path, sep="\t", index=False)
    resolution = resolve_group_labels(
        retained,
        samples,
        preferred_column="SUBJECT_ID",
        source_metadata_path=path,
        auto_detect=True,
        namespace_columns=["location"],
    )
    assert resolution.groups.notna().all()
    assert resolution.groups.nunique() == 4
    assert resolution.info["identifier_columns_used"] == {"SUBJECT_ID2": 4, "SubjectID": 4}
