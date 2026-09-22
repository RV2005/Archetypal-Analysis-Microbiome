"""PCoA, UMAP, and t-SNE helpers, including supplementary archetype projection."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class ProjectedEmbedding:
    """Sample coordinates plus supplementary coordinates in the same fitted space."""

    sample_coords: pd.DataFrame
    supplemental_coords: pd.DataFrame
    extra: Any = None
    projection_method: str | None = None


def preprocess_embedding_data(otu: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    x = otu.copy().astype(float)
    if cfg.get("log_transform", True):
        base = float(cfg.get("log_base", 2.0))
        pseudocount = float(cfg.get("pseudocount", 1.0))
        x = np.log(x + pseudocount) / np.log(base)
    x = x.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if cfg.get("standard_scale", False):
        x.loc[:, :] = StandardScaler().fit_transform(x)
    return x


def _preprocess_reference_and_supplemental(
    reference: pd.DataFrame,
    supplemental: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one embedding transform learned from samples to samples + archetypes."""
    columns = list(reference.columns)
    ref = reference.loc[:, columns].copy().astype(float)
    sup = supplemental.reindex(columns=columns).copy().astype(float)

    # Missing supplemental features correspond to features absent from the fitted AA
    # representation (normally zero-variance features removed before fit). Fill them
    # from the reference value so they do not create an artificial displacement.
    missing = sup.isna().all(axis=0)
    if bool(missing.any()) and len(ref):
        sup.loc[:, missing] = ref.iloc[0].loc[missing].to_numpy()

    if cfg.get("log_transform", True):
        base = float(cfg.get("log_base", 2.0))
        pseudocount = float(cfg.get("pseudocount", 1.0))
        ref = np.log(ref + pseudocount) / np.log(base)
        sup = np.log(sup + pseudocount) / np.log(base)

    ref = ref.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    sup = sup.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    if cfg.get("standard_scale", False):
        scaler = StandardScaler().fit(ref)
        ref.loc[:, :] = scaler.transform(ref)
        sup.loc[:, :] = scaler.transform(sup)
    return ref, sup


def _natural_key(value: Any):
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", str(value))]


def plot_embedding(
    coords: pd.DataFrame,
    metadata: pd.DataFrame,
    target: str,
    legend_type: str,
    cfg: Mapping[str, Any],
    title: str,
    color_map: Mapping[str, Any] | None = None,
):
    common = coords.index.intersection(metadata.index, sort=False)
    coords = coords.loc[common]
    labels = metadata.loc[common, target]
    valid = labels.notna()
    coords = coords.loc[valid]
    labels = labels.loc[valid]

    figsize = tuple(cfg.get("figsize", [7, 6]))
    fig, ax = plt.subplots(figsize=figsize)
    cmap_name = cfg.get("cmap", "tab20" if legend_type == "discrete" else "viridis")
    alpha = float(cfg.get("alpha", 0.7))
    size = float(cfg.get("s", 18))

    if legend_type == "continuous":
        numeric = pd.to_numeric(labels, errors="coerce")
        finite = numeric.notna()
        scatter = ax.scatter(
            coords.loc[finite, coords.columns[0]],
            coords.loc[finite, coords.columns[1]],
            c=numeric.loc[finite],
            cmap=cmap_name,
            alpha=alpha,
            s=size,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(cfg.get("colorbar_title", target))
    else:
        categories = sorted(labels.astype(str).unique(), key=_natural_key)
        cmap = plt.get_cmap(cmap_name, max(1, len(categories)))
        colors = [color_map.get(label, cmap(i)) if color_map else cmap(i) for i, label in enumerate(categories)]
        label_to_color = dict(zip(categories, colors))
        point_colors = labels.astype(str).map(label_to_color)
        ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=list(point_colors), alpha=alpha, s=size)
        handles = [Patch(color=label_to_color[label], label=label) for label in categories]
        ax.legend(
            handles=handles,
            title=target,
            loc=cfg.get("legend_loc", "best"),
            fontsize=cfg.get("legend_fontsize", 9),
        )

    ax.set_xlabel(coords.columns[0])
    ax.set_ylabel(coords.columns[1])
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def _umap_model(cfg: Mapping[str, Any], n_samples: int):
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP plotting requires 'umap-learn'. Install the project dependencies.") from exc
    n_neighbors = min(int(cfg.get("n_neighbors", 15)), max(2, n_samples - 1))
    return umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=int(cfg.get("n_components", 2)),
        min_dist=float(cfg.get("min_dist", 0.1)),
        metric=str(cfg.get("metric", "euclidean")),
        random_state=int(cfg.get("random_state", 3407)),
        **dict(cfg.get("kwargs", {})),
    )


def compute_umap(otu: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    x = preprocess_embedding_data(otu, cfg)
    model = _umap_model(cfg, len(x))
    embedding = model.fit_transform(x.to_numpy())
    return pd.DataFrame(embedding[:, :2], index=x.index, columns=["UMAP1", "UMAP2"])


def compute_umap_projection(
    otu: pd.DataFrame,
    supplemental: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> ProjectedEmbedding:
    """Fit UMAP on samples and project archetypes into that fixed sample space.

    By default we avoid UMAP's native ``transform`` because it can segfault in some
    Python/Numba/OpenMP environments during pipeline plotting. Instead we keep the
    sample embedding exact and project supplementary archetypes with distance-weighted
    K-nearest-neighbor regression from the preprocessed input space to the learned
    UMAP coordinates. Users can opt back into the native transform with
    ``archetype_projection.method: native_transform`` if their environment is stable.
    """
    x, x_sup = _preprocess_reference_and_supplemental(otu, supplemental, cfg)
    model = _umap_model(cfg, len(x))
    sample_embedding = model.fit_transform(x.to_numpy())
    sample_coords = pd.DataFrame(sample_embedding[:, :2], index=x.index, columns=["UMAP1", "UMAP2"])

    projection_cfg = cfg.get("archetype_projection", {}) or {}
    method = str(projection_cfg.get("method", "knn")).lower()
    if method == "native_transform":
        supplemental_embedding = model.transform(x_sup.to_numpy()) if len(x_sup) else np.empty((0, 2))
        projection_method = "umap_native_transform"
    elif method == "knn":
        n_neighbors = min(int(projection_cfg.get("n_neighbors", 10)), len(x))
        weights = projection_cfg.get("weights", "distance")
        metric = str(projection_cfg.get("metric", cfg.get("metric", "euclidean")))
        regressor = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)
        regressor.fit(x.to_numpy(), sample_embedding[:, :2])
        supplemental_embedding = regressor.predict(x_sup.to_numpy()) if len(x_sup) else np.empty((0, 2))
        projection_method = f"umap_knn_{n_neighbors}_{weights}"
    else:
        raise ValueError(
            "Unsupported UMAP archetype_projection.method. "
            "Use 'knn' (default) or 'native_transform'."
        )
    return ProjectedEmbedding(
        sample_coords,
        pd.DataFrame(supplemental_embedding[:, :2], index=x_sup.index, columns=["UMAP1", "UMAP2"]),
        extra=model,
        projection_method=projection_method,
    )


def _tsne_model(cfg: Mapping[str, Any], n_samples: int) -> TSNE:
    perplexity = float(cfg.get("perplexity", 30.0))
    if perplexity >= n_samples:
        raise ValueError(f"t-SNE perplexity ({perplexity}) must be smaller than n_samples ({n_samples})")
    kwargs = dict(cfg.get("kwargs", {}))
    tsne_args = {
        "n_components": int(cfg.get("n_components", 2)),
        "perplexity": perplexity,
        "learning_rate": cfg.get("learning_rate", "auto"),
        "init": str(cfg.get("init", "pca")),
        "metric": str(cfg.get("metric", "euclidean")),
        "random_state": int(cfg.get("random_state", 3407)),
        **kwargs,
    }
    iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    tsne_args[iteration_key] = int(cfg.get("max_iter", 1000))
    return TSNE(**tsne_args)


def compute_tsne(otu: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    x = preprocess_embedding_data(otu, cfg)
    model = _tsne_model(cfg, len(x))
    embedding = model.fit_transform(x.to_numpy())
    return pd.DataFrame(embedding[:, :2], index=x.index, columns=["tSNE1", "tSNE2"])


def compute_tsne_projection(
    otu: pd.DataFrame,
    supplemental: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> ProjectedEmbedding:
    """Fit t-SNE on samples and approximate supplementary points with local regression.

    scikit-learn t-SNE has no out-of-sample ``transform`` method. We therefore keep
    all sample coordinates exactly in the fitted t-SNE space and project archetypes by
    distance-weighted K-nearest-neighbor regression from the preprocessed input space
    to the learned two-dimensional t-SNE coordinates.
    """
    x, x_sup = _preprocess_reference_and_supplemental(otu, supplemental, cfg)
    model = _tsne_model(cfg, len(x))
    sample_embedding = model.fit_transform(x.to_numpy())
    sample_coords = pd.DataFrame(sample_embedding[:, :2], index=x.index, columns=["tSNE1", "tSNE2"])

    projection_cfg = cfg.get("archetype_projection", {}) or {}
    n_neighbors = min(int(projection_cfg.get("n_neighbors", 10)), len(x))
    weights = projection_cfg.get("weights", "distance")
    metric = str(projection_cfg.get("metric", cfg.get("metric", "euclidean")))
    regressor = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)
    regressor.fit(x.to_numpy(), sample_embedding[:, :2])
    supplemental_embedding = regressor.predict(x_sup.to_numpy()) if len(x_sup) else np.empty((0, 2))
    return ProjectedEmbedding(
        sample_coords,
        pd.DataFrame(supplemental_embedding, index=x_sup.index, columns=["tSNE1", "tSNE2"]),
        extra=regressor,
        projection_method=f"knn_{n_neighbors}_{weights}",
    )


def project_archetypes_from_similarity(
    projection: ProjectedEmbedding,
    results: Mapping[int, Mapping[str, Any]],
    archetype_ids: Mapping[int, Mapping[str, str]],
    cfg: Mapping[str, Any],
) -> ProjectedEmbedding:
    """Place archetypes in nonlinear embeddings by their AA sample affinities."""
    projection_cfg = cfg.get("archetype_projection", {}) or {}
    power = float(projection_cfg.get("similarity_power", 2.0))
    if power <= 0:
        raise ValueError("archetype_projection.similarity_power must be > 0")

    coords = projection.sample_coords
    rows: list[pd.Series] = []
    for k, result in sorted(results.items()):
        sim = pd.DataFrame(
            result["similarity_degree"],
            index=pd.Index(result["index"], name="sample"),
            columns=result["columns"],
        )
        common = sim.index.intersection(coords.index, sort=False)
        if len(common) == 0:
            continue
        sim = sim.loc[common]
        coord_values = coords.loc[common].to_numpy(dtype=float)

        for archetype_name, row_id in archetype_ids.get(int(k), {}).items():
            if archetype_name not in sim.columns:
                continue
            weights = np.clip(sim[archetype_name].to_numpy(dtype=float), a_min=0.0, a_max=None)
            if not np.isclose(power, 1.0):
                weights = np.power(weights, power)
            total = float(weights.sum())
            if not np.isfinite(total) or total <= 0:
                values = np.full(len(coords.columns), np.nan, dtype=float)
            else:
                values = (coord_values * weights[:, None]).sum(axis=0) / total
            rows.append(pd.Series(values, index=coords.columns, name=row_id))

    supplemental = pd.DataFrame(rows, columns=coords.columns)
    return ProjectedEmbedding(
        sample_coords=projection.sample_coords,
        supplemental_coords=supplemental,
        extra=projection.extra,
        projection_method=f"{projection.projection_method}+similarity_centroid_power_{power:g}",
    )


def compute_pcoa(otu: pd.DataFrame, cfg: Mapping[str, Any]):
    from .pcoa import PCoAVisualizer

    visualizer = PCoAVisualizer()
    return visualizer.create_pcoa_from_otu(
        otu_data=otu,
        distance_metric=str(cfg.get("distance_metric", "jensenshannon")),
        n_dimensions=int(cfg.get("n_dimensions", 3)),
        apply_log_transform=bool(cfg.get("log_transform", True)),
        clean_data=bool(cfg.get("clean_data", True)),
        verbose=bool(cfg.get("verbose", False)),
    )


def _preprocess_pcoa_data(data: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    from ..data.formatters import DataTransformer

    x = data.copy().astype(float)
    if cfg.get("log_transform", True):
        x = np.log2(x + 1)
    if cfg.get("clean_data", True):
        x = DataTransformer.general_cleanup(x)
    return x


def _distance_metric(cfg: Mapping[str, Any]):
    metric = str(cfg.get("distance_metric", "jensenshannon"))
    return distance.jensenshannon if metric == "jensenshannon" else metric


def compute_pcoa_projection(
    otu: pd.DataFrame,
    supplemental: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> ProjectedEmbedding:
    """Project archetypes as supplementary points into the sample-only PCoA space.

    The sample PCoA is unchanged. Supplementary coordinates use classical-MDS/Gower
    interpolation from distances between each archetype and the fitted samples.
    """
    sample_coords, pcoa_results = compute_pcoa(otu, cfg)
    reference = _preprocess_pcoa_data(otu, cfg).reindex(sample_coords.index)
    supplemental_processed = _preprocess_pcoa_data(supplemental, cfg)
    supplemental_processed = supplemental_processed.reindex(columns=reference.columns)
    supplemental_processed = supplemental_processed.dropna(axis=0, how="any")

    metric = _distance_metric(cfg)
    train_values = reference.to_numpy(dtype=float)
    train_dist = distance.squareform(distance.pdist(train_values, metric=metric))
    squared = train_dist ** 2
    row_mean = squared.mean(axis=1)
    grand_mean = float(squared.mean())

    eigvals = pd.Series(pcoa_results.eigvals).reindex(sample_coords.columns)
    projected = np.full((len(supplemental_processed), len(sample_coords.columns)), np.nan, dtype=float)
    if len(supplemental_processed):
        cross = distance.cdist(supplemental_processed.to_numpy(dtype=float), train_values, metric=metric)
        for j, column in enumerate(sample_coords.columns):
            eigenvalue = float(eigvals.loc[column]) if column in eigvals.index else np.nan
            if not np.isfinite(eigenvalue) or eigenvalue <= 1e-12:
                continue
            eigenvector = sample_coords[column].to_numpy(dtype=float) / np.sqrt(eigenvalue)
            for i, d in enumerate(cross):
                d2 = d ** 2
                centered_inner_product = -0.5 * (d2 - row_mean - float(d2.mean()) + grand_mean)
                projected[i, j] = float(centered_inner_product @ eigenvector / np.sqrt(eigenvalue))

    supplemental_coords = pd.DataFrame(
        projected,
        index=supplemental_processed.index,
        columns=sample_coords.columns,
    )
    return ProjectedEmbedding(
        sample_coords,
        supplemental_coords,
        extra=pcoa_results,
        projection_method="gower_supplementary_projection",
    )


def compute_projection(
    kind: str,
    otu: pd.DataFrame,
    supplemental: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> ProjectedEmbedding:
    kind = str(kind).lower()
    if kind == "pcoa":
        return compute_pcoa_projection(otu, supplemental, cfg)
    if kind == "umap":
        return compute_umap_projection(otu, supplemental, cfg)
    if kind == "tsne":
        return compute_tsne_projection(otu, supplemental, cfg)
    raise ValueError(f"Unknown embedding kind: {kind}")