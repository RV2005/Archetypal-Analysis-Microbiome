import numpy as np
import pandas as pd

from archetypal_analysis.visualization import embeddings


class _FakeUMAP:
    def fit_transform(self, x):
        x = np.asarray(x, dtype=float)
        return np.column_stack((x[:, 0], x[:, 1]))

    def transform(self, x):
        raise AssertionError("native_transform should not be used by default")


def test_umap_projection_defaults_to_knn_without_native_transform(monkeypatch):
    monkeypatch.setattr(embeddings, "_umap_model", lambda cfg, n_samples: _FakeUMAP())

    otu = pd.DataFrame(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        index=["s1", "s2", "s3"],
        columns=["f1", "f2"],
    )
    supplemental = pd.DataFrame([[1.5, 1.5]], index=["a1"], columns=["f1", "f2"])

    projected = embeddings.compute_umap_projection(
        otu,
        supplemental,
        {"log_transform": False, "archetype_projection": {}},
    )

    assert projected.projection_method == "umap_knn_3_distance"
    assert list(projected.sample_coords.columns) == ["UMAP1", "UMAP2"]
    assert list(projected.supplemental_coords.index) == ["a1"]
    assert projected.supplemental_coords.shape == (1, 2)


def test_similarity_projection_separates_distinct_archetype_affinities():
    projection = embeddings.ProjectedEmbedding(
        sample_coords=pd.DataFrame(
            [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
            index=["s1", "s2", "s3", "s4"],
            columns=["UMAP1", "UMAP2"],
        ),
        supplemental_coords=pd.DataFrame(),
        projection_method="umap_knn_4_distance",
    )
    result = {
        "similarity_degree": [
            [0.90, 0.05],
            [0.85, 0.10],
            [0.10, 0.80],
            [0.05, 0.90],
        ],
        "index": ["s1", "s2", "s3", "s4"],
        "columns": ["Archetype 1", "Archetype 2"],
    }

    projected = embeddings.project_archetypes_from_similarity(
        projection,
        {2: result},
        {2: {"Archetype 1": "K_002::Archetype 1", "Archetype 2": "K_002::Archetype 2"}},
        {"archetype_projection": {"similarity_power": 2.0}},
    )

    first = projected.supplemental_coords.loc["K_002::Archetype 1"]
    second = projected.supplemental_coords.loc["K_002::Archetype 2"]
    assert first["UMAP2"] < second["UMAP2"]
    assert not np.allclose(first.to_numpy(), second.to_numpy())
    assert projected.projection_method == "umap_knn_4_distance+similarity_centroid_power_2"
