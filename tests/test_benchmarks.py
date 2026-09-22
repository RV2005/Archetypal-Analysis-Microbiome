from pathlib import Path

import numpy as np
import pandas as pd

from archetypal_analysis.analysis.benchmarks import benchmark_tasks, compute_benchmark


def test_benchmark_task_contract_fits_linear_once_and_nonlinear_per_dimension():
    cfg = {
        "enabled": True,
        "pca": {"enabled": True},
        "pcoa": {"enabled": True},
        "umap": {"enabled": True},
        "tsne": {"enabled": True},
    }
    tasks = benchmark_tasks([2, 4, 5], cfg)
    pairs = [(task.method, task.dimension) for task in tasks]
    # AA K values [2, 4, 5] have intrinsic dimensions [1, 3, 4].
    assert pairs.count(("pca", 4)) == 1
    assert pairs.count(("pcoa", 4)) == 1
    assert [(m, d) for m, d in pairs if m == "umap"] == [("umap", 1), ("umap", 3), ("umap", 4)]
    assert [(m, d) for m, d in pairs if m == "tsne"] == [("tsne", 1), ("tsne", 3), ("tsne", 4)]


def test_pca_and_pcoa_benchmarks_return_requested_max_dimensions():
    rng = np.random.RandomState(4)
    data = pd.DataFrame(
        rng.normal(size=(20, 8)),
        index=[f"s{i}" for i in range(20)],
        columns=[f"f{i}" for i in range(8)],
    )
    pca, pca_meta, extras = compute_benchmark(data, "pca", 5, {"random_state": 4})
    assert pca.shape == (20, 5)
    assert pca_meta["dimension"] == 5
    assert {"components.tsv", "mean.tsv", "explained_variance.tsv"}.issubset(extras)

    pcoa, pcoa_meta, pcoa_extras = compute_benchmark(data, "pcoa", 5, {"distance_metric": "euclidean"})
    assert pcoa.shape == (20, 5)
    assert pcoa_meta["distance_metric"] == "euclidean"
    assert "eigenvalues.tsv" in pcoa_extras
