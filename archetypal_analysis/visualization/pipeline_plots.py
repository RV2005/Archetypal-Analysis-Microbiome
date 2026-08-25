"""High-level plots produced by the config-driven pipeline."""

from __future__ import annotations

import math
import re
import colorsys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..analysis.experiments import inverse_archetypes, similarity_frame
from .archetype_ordering import clean_task_labels, natural_key, resolve_archetype_order
from .embeddings import ProjectedEmbedding, compute_pcoa, compute_tsne, compute_umap, plot_embedding
from .simplex import SimplexPlotter
from .stacked_bar import StackedBarPlotter


def _clean_labels(metadata: pd.DataFrame, task: Mapping[str, Any]) -> pd.DataFrame:
    return clean_task_labels(metadata, task)


def _archetype_order(
    result: Mapping[str, Any],
    order: Sequence[str] | None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    sim = similarity_frame(result)
    arch = np.asarray(result["archetypes"], dtype=float)
    names = list(sim.columns)
    if order and len(order) == len(names) and set(order) == set(names):
        idx = [names.index(name) for name in order]
        sim = sim.loc[:, list(order)]
        arch = arch[idx]
        names = list(order)
    return sim, arch, names


def relabel_result_for_visual_order(
    result: Mapping[str, Any],
    order: Sequence[str],
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    """Return a visualization-only result ordered then renamed as Archetype 1..K."""
    sim = similarity_frame(result)
    names = list(sim.columns)
    if not order or len(order) != len(names) or set(order) != set(names):
        order = names
    order = list(order)
    display_names = [f"Archetype {i + 1}" for i in range(len(order))]
    original_to_display = dict(zip(order, display_names))
    idx = [names.index(name) for name in order]

    visual_result = dict(result)
    visual_result["columns"] = display_names
    visual_result["similarity_degree"] = sim.loc[:, order].to_numpy().tolist()

    if "archetypes" in result:
        archetypes = np.asarray(result["archetypes"], dtype=float)
        visual_result["archetypes"] = archetypes[idx].tolist()

    # Saved permutations refer to the original vertex order and are no longer valid
    # once columns are explicitly reordered for display.
    visual_result["info_perms"] = []
    return visual_result, display_names, original_to_display


def save_figure(fig, path: Path, cfg: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(cfg.get("dpi", 180)), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metrics(metrics: pd.DataFrame, cfg: Mapping[str, Any]):
    fig, ax1 = plt.subplots(figsize=tuple(cfg.get("figsize", [9, 5])))
    ax2 = ax1.twinx()
    l1 = ax1.plot(
        metrics["k"],
        metrics["silhouette"],
        marker=cfg.get("silhouette_marker", "o"),
        linewidth=float(cfg.get("linewidth", 2)),
        label="Silhouette",
    )
    l2 = ax2.plot(
        metrics["k"],
        metrics["davies_bouldin"],
        marker=cfg.get("davies_marker", "s"),
        linewidth=float(cfg.get("linewidth", 2)),
        label="Davies-Bouldin",
    )
    ax1.set_xlabel("Number of archetypes (K)")
    ax1.set_ylabel("Silhouette score")
    ax2.set_ylabel("Davies-Bouldin score")
    ax1.set_title(cfg.get("title", "Silhouette and Davies-Bouldin over K"))
    ax1.grid(True, alpha=float(cfg.get("grid_alpha", 0.25)))
    lines = l1 + l2
    ax1.legend(lines, [line.get_label() for line in lines], loc=cfg.get("legend_loc", "best"))
    fig.tight_layout()
    return fig


def plot_quality_selection(metrics: pd.DataFrame, cfg: Mapping[str, Any]):
    fig, ax = plt.subplots(figsize=tuple(cfg.get("figsize", [9, 4.5])))
    ax.plot(
        metrics["k"],
        metrics["quality_score"],
        marker="o",
        label="normalized silhouette - normalized DB",
    )
    ax.set_xlabel("Number of archetypes (K)")
    ax.set_ylabel("Composite score")
    ax.set_title(cfg.get("title", "Normalized quality score over K"))
    ax.grid(True, alpha=float(cfg.get("grid_alpha", 0.25)))
    ax.legend()
    fig.tight_layout()
    return fig


def _draw_simplex(
    ax,
    result: Mapping[str, Any],
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
    cfg: Mapping[str, Any],
    order: Sequence[str] | None,
):
    metadata = _clean_labels(metadata, task)
    sim, _, _ = _archetype_order(result, order)
    SimplexPlotter().plot_simplex(
        otu_aa=sim,
        metadata=metadata,
        filter_cols=[task["target"]],
        cmap_name=cfg.get("cmap", "tab20" if task.get("legend_type") == "discrete" else "viridis"),
        legend_type=task.get("legend_type", "discrete"),
        remove_nan=bool(cfg.get("remove_nan", True)),
        remove_values={task["target"]: value for value in (cfg.get("remove_values", []) or [])},
        alpha=float(cfg.get("alpha", 0.75)),
        s=float(cfg.get("s", 25)),
        ax=ax,
        legend_fontsize=int(cfg.get("legend_fontsize", 9)),
        custom_colors=task.get("color_map"),
        make_arrows=bool(cfg.get("make_arrows", False)),
    )
    return ax


def plot_single_simplex(
    result: dict[str, Any],
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
    cfg: Mapping[str, Any],
    order: Sequence[str] | None = None,
):
    order = list(order) if order is not None else cfg.get("archetype_order")
    fig, ax = plt.subplots(figsize=tuple(cfg.get("figsize", [8, 8])))
    _draw_simplex(ax, result, metadata, task, cfg, order)
    ax.set_title(cfg.get("title", f"Archetypal simplex (K={result['k']})"))
    fig.tight_layout()
    return fig


def plot_multi_k_simplex(
    results: Mapping[int, dict[str, Any]],
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
    cfg: Mapping[str, Any],
):
    ks = sorted(results)
    ncols = max(1, int(cfg.get("ncols", 2)))
    nrows = math.ceil(len(ks) / ncols)
    panel = cfg.get("panel_size", [6, 6])
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(float(panel[0]) * ncols, float(panel[1]) * nrows),
        squeeze=False,
    )
    for ax, k in zip(axes.flat, ks):
        result = results[k]
        order = resolve_archetype_order(result, metadata, task, cfg)
        visual_result, display_order, _ = relabel_result_for_visual_order(result, order)
        _draw_simplex(ax, visual_result, metadata, task, cfg, display_order)
        ax.set_title(f"K={k}")
    for ax in list(axes.flat)[len(ks) :]:
        ax.set_visible(False)
    fig.suptitle(cfg.get("title", "Simplex comparison across K"))
    fig.tight_layout()
    return fig


def _draw_similarity_stacked(
    ax,
    result: Mapping[str, Any],
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
    cfg: Mapping[str, Any],
    order: Sequence[str] | None,
):
    metadata = _clean_labels(metadata, task)
    sim, _, names = _archetype_order(result, order)
    common = sim.index.intersection(metadata.index, sort=False)
    sim = sim.loc[common]
    plot_metadata = metadata.loc[common]
    perms = result.get("info_perms") or None
    if order:
        # Saved sample permutations were computed in the original vertex ordering.
        perms = None
    buckets = cfg.get("buckets")
    if buckets is not None:
        buckets = [(lo, float("inf") if hi is None else hi) for lo, hi in buckets]
    labels = plot_metadata[task["target"]].values if cfg.get("group_by_target", True) else None
    StackedBarPlotter().stacked_bar(
        points=sim.to_numpy(),
        perms=perms,
        ax=ax,
        labels=labels,
        discretize=bool(cfg.get("discretize", False)),
        buckets=buckets,
        vertex_colors=cfg.get("archetype_colors"),
        vertices_labels=names,
        cmap_name=cfg.get("cmap", "tab20"),
        alpha=float(cfg.get("alpha", 0.85)),
    )
    return ax


def plot_similarity_stacked(
    result: dict[str, Any],
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
    cfg: Mapping[str, Any],
    order: Sequence[str] | None = None,
):
    order = list(order) if order is not None else cfg.get("archetype_order")
    fig, ax = plt.subplots(figsize=tuple(cfg.get("figsize", [15, 5])))
    _draw_similarity_stacked(ax, result, metadata, task, cfg, order)
    ax.set_title(cfg.get("title", f"Archetype similarity composition (K={result['k']})"))
    fig.tight_layout()
    return fig


def _simplify_taxon(name: str, level: int = -1, max_length: int = 30) -> str:
    parts = [
        part
        for part in str(name).split(".")
        if part and part not in {"NA", "nan", "unknown", "unclassified"}
    ]
    if not parts:
        value = str(name)
    elif level == -1 or not (0 <= level < len(parts)):
        value = parts[-1]
    else:
        value = parts[level]
    return value if len(value) <= max_length else value[:max_length] + "..."


def plot_taxa_bars(result: dict[str, Any], transform: Mapping[str, Any], cfg: Mapping[str, Any]):
    """Legacy API retained for notebooks; the pipeline no longer emits taxa_bars.png."""
    arch = inverse_archetypes(np.asarray(result["archetypes"], dtype=float), transform)
    features = list(result["feature_names"])
    n = int(cfg.get("num_taxa", 5))
    names = [f"Archetype {i + 1}" for i in range(len(arch))]
    ncols = max(1, int(cfg.get("ncols", min(3, len(arch)))))
    nrows = math.ceil(len(arch) / ncols)
    panel = cfg.get("panel_size", [5, 4])
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(float(panel[0]) * ncols, float(panel[1]) * nrows),
        squeeze=False,
    )
    level = int(cfg.get("simplified_taxa_level", -1))
    max_len = int(cfg.get("max_name_length", 30))
    for ax, values, arch_name in zip(axes.flat, arch, names):
        ranked = np.argsort(values)[::-1][:n]
        total = values.sum()
        fractions = values[ranked] / total if total > 0 else np.zeros(len(ranked))
        labels = [_simplify_taxon(features[j], level, max_len) for j in ranked]
        ax.bar(labels, fractions)
        ax.set_title(arch_name)
        ax.set_ylabel("Fraction")
        ax.tick_params(axis="x", labelrotation=float(cfg.get("label_rotation", 30)))
    for ax in list(axes.flat)[len(arch) :]:
        ax.set_visible(False)
    fig.suptitle(cfg.get("title", f"Top taxa by archetype (K={result['k']})"))
    fig.tight_layout()
    return fig


def _taxa_stacked_data(
    result: Mapping[str, Any],
    transform: Mapping[str, Any],
    cfg: Mapping[str, Any],
    order: Sequence[str] | None,
) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    arch = inverse_archetypes(np.asarray(result["archetypes"], dtype=float), transform)
    features = list(result["feature_names"])
    names = [f"Archetype {i + 1}" for i in range(len(arch))]
    if order and len(order) == len(names) and set(order) == set(names):
        idx = [names.index(name) for name in order]
        arch = arch[idx]
        names = list(order)
    n = int(cfg.get("num_taxa", 5))
    top = sorted(set(np.concatenate([np.argsort(row)[::-1][:n] for row in arch]).tolist()))
    totals = arch.sum(axis=1)
    fractions = np.divide(
        arch[:, top],
        totals[:, None],
        out=np.zeros((len(arch), len(top))),
        where=totals[:, None] > 0,
    )
    return fractions, names, top, features


def _taxa_colors(n_colors: int):
    if n_colors <= 12:
        cmap = plt.cm.Set3
        return [cmap(i / max(1, 12)) for i in range(n_colors)]
    if n_colors <= 20:
        cmap = plt.cm.tab20
        return [cmap(i / max(1, 20)) for i in range(n_colors)]
    colors = []
    for i in range(n_colors):
        hue = i / max(1, n_colors)
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.7 + (i % 2) * 0.2
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return colors


def visual_top_taxa_frame(
    result: Mapping[str, Any],
    transform: Mapping[str, Any],
    cfg: Mapping[str, Any],
    order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Top taxa table using the same archetype labels/order as the visualization."""
    fractions, names, top, features = _taxa_stacked_data(result, transform, cfg, order)
    n = int(cfg.get("num_taxa", 5))
    rows = []
    for arch_index, archetype in enumerate(names):
        ranked_columns = np.argsort(fractions[arch_index])[::-1][:n]
        for rank, column in enumerate(ranked_columns, 1):
            value = float(fractions[arch_index, column])
            if value <= 0:
                continue
            feature_index = top[int(column)]
            rows.append(
                {
                    "archetype": archetype,
                    "rank": rank,
                    "taxon": features[feature_index],
                    "display_taxon": _simplify_taxon(
                        features[feature_index],
                        int(cfg.get("simplified_taxa_level", -1)),
                        int(cfg.get("max_name_length", 30)),
                    ),
                    "fraction": value,
                }
            )
    return pd.DataFrame(rows)


def _draw_taxa_stacked(
    ax,
    result: Mapping[str, Any],
    transform: Mapping[str, Any],
    cfg: Mapping[str, Any],
    order: Sequence[str] | None,
):
    fractions, names, top, features = _taxa_stacked_data(result, transform, cfg, order)
    bottom = np.zeros(len(names))
    level = int(cfg.get("simplified_taxa_level", -1))
    max_len = int(cfg.get("max_name_length", 30))
    colors = _taxa_colors(len(top))
    for column, feature_index in enumerate(top):
        label = _simplify_taxon(features[feature_index], level, max_len)
        ax.bar(
            np.arange(len(names)),
            fractions[:, column],
            bottom=bottom,
            label=label,
            color=colors[column],
        )
        bottom += fractions[:, column]
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Fraction of archetype")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return ax


def plot_taxa_stacked(
    result: dict[str, Any],
    transform: Mapping[str, Any],
    cfg: Mapping[str, Any],
    order: Sequence[str] | None = None,
):
    fig, ax = plt.subplots(figsize=tuple(cfg.get("stacked_figsize", [10, 6])))
    _draw_taxa_stacked(ax, result, transform, cfg, order)
    ax.set_title(cfg.get("stacked_title", f"Top-taxa composition (K={result['k']})"))
    fig.tight_layout()
    return fig


def plot_pcoa_dataset(
    otu: pd.DataFrame,
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
    cfg: Mapping[str, Any],
):
    coords, pcoa_results = compute_pcoa(otu, cfg)
    metadata = _clean_labels(metadata, task)
    pc_combos = [
        tuple(x)
        for x in cfg.get("pc_combinations", [["PC1", "PC2"], ["PC2", "PC3"], ["PC1", "PC3"]])
    ]
    fig, _axes = PCoAPlotProxy.plot(coords, pcoa_results, metadata, task, cfg, pc_combos)
    return fig


class PCoAPlotProxy:
    @staticmethod
    def plot(coords, results, metadata, task, cfg, pc_combos):
        from .pcoa import PCoAVisualizer

        fig, axes = PCoAVisualizer().plot_pcoa(
            pcoa_coordinates=coords,
            pcoa_results=results,
            metadata=metadata,
            filter_col=task["target"],
            legend_type=task.get("legend_type", "discrete"),
            figsize=tuple(cfg.get("figsize", [18, 6])),
            alpha=float(cfg.get("alpha", 0.65)),
            s=float(cfg.get("s", 14)),
            pc_combinations=pc_combos,
            cmap_name=cfg.get("cmap", "tab20" if task.get("legend_type") == "discrete" else "viridis"),
        )
        if task.get("legend_type") == "continuous" and len(fig.axes) > len(pc_combos):
            fig.axes[-1].set_ylabel(task.get("colorbar_title", task["target"]))
        fig.suptitle(cfg.get("title", "PCoA of fitted dataset"))
        fig.tight_layout()
        return fig, axes


def plot_umap_dataset(otu, metadata, task, cfg):
    coords = compute_umap(otu, cfg)
    local = dict(cfg)
    local.setdefault("colorbar_title", task.get("colorbar_title", task["target"]))
    return plot_embedding(
        coords,
        _clean_labels(metadata, task),
        task["target"],
        task.get("legend_type", "discrete"),
        local,
        cfg.get("title", "UMAP of fitted dataset"),
        task.get("color_map"),
    )[0]


def plot_tsne_dataset(otu, metadata, task, cfg):
    coords = compute_tsne(otu, cfg)
    local = dict(cfg)
    local.setdefault("colorbar_title", task.get("colorbar_title", task["target"]))
    return plot_embedding(
        coords,
        _clean_labels(metadata, task),
        task["target"],
        task.get("legend_type", "discrete"),
        local,
        cfg.get("title", "t-SNE of fitted dataset"),
        task.get("color_map"),
    )[0]


def build_supplemental_archetypes(
    results: Mapping[int, Mapping[str, Any]],
    transform: Mapping[str, Any],
    reference_otu: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, dict[str, str]]]:
    """Convert all selected-K archetypes to raw-like supplementary rows once."""
    rows: list[pd.DataFrame] = []
    ids_by_k: dict[int, dict[str, str]] = {}
    reference_columns = list(reference_otu.columns)
    reference_fill = reference_otu.iloc[0] if len(reference_otu) else pd.Series(0.0, index=reference_columns)

    for k, result in sorted(results.items()):
        values = inverse_archetypes(np.asarray(result["archetypes"], dtype=float), transform)
        fitted_features = list(result["feature_names"])
        names = [f"Archetype {i + 1}" for i in range(len(values))]
        frame = pd.DataFrame(index=names, columns=reference_columns, dtype=float)
        present = [feature for feature in fitted_features if feature in frame.columns]
        if present:
            source_indices = [fitted_features.index(feature) for feature in present]
            frame.loc[:, present] = values[:, source_indices]
        missing = frame.columns[frame.isna().all(axis=0)]
        if len(missing):
            frame.loc[:, missing] = reference_fill.loc[missing].to_numpy()
        frame = frame.fillna(reference_fill)

        id_map: dict[str, str] = {}
        renamed = []
        for name in names:
            row_id = f"K_{int(k):03d}::{name}"
            id_map[name] = row_id
            renamed.append(row_id)
        frame.index = renamed
        rows.append(frame)
        ids_by_k[int(k)] = id_map

    if not rows:
        return pd.DataFrame(columns=reference_columns), ids_by_k
    return pd.concat(rows, axis=0), ids_by_k


def _resolve_archetype_color(name: str, cfg: Mapping[str, Any], index: int):
    mapping = cfg.get("archetype_colors") or {}
    value = mapping.get(name)
    cmap = plt.get_cmap(cfg.get("archetype_cmap", "tab10"))
    if value is None:
        return cmap(index % max(1, cmap.N))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return cmap(float(value))
    return value


def _archetype_number(name: str, fallback: int) -> str:
    match = re.search(r"(\d+)$", str(name))
    return match.group(1) if match else str(fallback + 1)


def _use_numeric_marker(cfg: Mapping[str, Any]) -> bool:
    projection_cfg = cfg.get("archetype_projection", {}) or {}
    mode = str(projection_cfg.get("marker_mode", "")).lower()
    return bool(projection_cfg.get("numeric_marker", False)) or mode == "numeric"


def _marker_for_archetype(name: str, cfg: Mapping[str, Any], index: int):
    projection_cfg = cfg.get("archetype_projection", {}) or {}
    if _use_numeric_marker(cfg):
        number = _archetype_number(name, index)
        return rf"$\mathbf{{{number}}}$"
    return projection_cfg.get("marker", "X")


def _marker_size_for_archetype(cfg: Mapping[str, Any]) -> float:
    projection_cfg = cfg.get("archetype_projection", {}) or {}
    marker_size = float(projection_cfg.get("marker_size", 180))
    if _use_numeric_marker(cfg):
        return float(projection_cfg.get("numeric_marker_size", max(marker_size, 260)))
    return marker_size


def _archetype_marker_handles(
    cfg: Mapping[str, Any],
    order: Sequence[str],
) -> list[Line2D]:
    projection_cfg = cfg.get("archetype_projection", {}) or {}
    legend_cfg = projection_cfg.get("marker_legend", {}) or {}
    handles: list[Line2D] = []
    for index, name in enumerate(order):
        color = _resolve_archetype_color(name, cfg, index)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=_marker_for_archetype(name, cfg, index),
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor=projection_cfg.get("edgecolor", "black"),
                markeredgewidth=float(projection_cfg.get("linewidth", 1.2)),
                color=color,
                label=name,
                markersize=float(legend_cfg.get("legend_marker_size", projection_cfg.get("legend_marker_size", 12))),
            )
        )
    return handles


def _add_archetype_marker_legend(fig, cfg: Mapping[str, Any], order: Sequence[str]) -> None:
    projection_cfg = cfg.get("archetype_projection", {}) or {}
    legend_cfg = projection_cfg.get("marker_legend", {}) or {}
    enabled = bool(legend_cfg.get("enabled", projection_cfg.get("marker_legend_enabled", False)))
    if not enabled or not order:
        return

    handles = _archetype_marker_handles(cfg, order)
    ncol_value = legend_cfg.get("ncol")
    ncol = len(handles) if ncol_value is None else int(ncol_value)
    legend = fig.legend(
        handles=handles,
        loc=legend_cfg.get("loc", "lower center"),
        bbox_to_anchor=tuple(legend_cfg.get("bbox_to_anchor", [0.5, 0.0])),
        ncol=max(1, ncol),
        frameon=bool(legend_cfg.get("frameon", False)),
        title=legend_cfg.get("title", None),
        fontsize=legend_cfg.get("fontsize", 9),
    )
    legend.set_zorder(20)
    bottom = float(legend_cfg.get("bottom_adjust", 0.12))
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, bottom))


def _overlay_archetypes(
    ax,
    coords: pd.DataFrame,
    id_map: Mapping[str, str],
    order: Sequence[str],
    cfg: Mapping[str, Any],
    x_col: str,
    y_col: str,
):
    projection_cfg = cfg.get("archetype_projection", {}) or {}
    marker_size = _marker_size_for_archetype(cfg)
    edgecolor = projection_cfg.get("edgecolor", "black")
    linewidth = float(projection_cfg.get("linewidth", 1.2))
    label_points = bool(projection_cfg.get("label", True))
    label_fontsize = int(projection_cfg.get("label_fontsize", 9))

    for index, name in enumerate(order):
        row_id = id_map.get(name)
        if row_id is None or row_id not in coords.index:
            continue
        x = float(coords.loc[row_id, x_col])
        y = float(coords.loc[row_id, y_col])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        color = _resolve_archetype_color(name, cfg, index)
        ax.scatter(
            [x],
            [y],
            marker=_marker_for_archetype(name, cfg, index),
            s=marker_size,
            c=[color],
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=8,
        )
        if label_points:
            ax.annotate(
                name,
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=label_fontsize,
                color=color,
                weight="bold",
                zorder=9,
            )


def plot_projected_embedding(
    projection: ProjectedEmbedding,
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
    cfg: Mapping[str, Any],
    id_map: Mapping[str, str],
    order: Sequence[str],
    title: str,
):
    local = dict(cfg)
    local.setdefault("colorbar_title", task.get("colorbar_title", task["target"]))
    fig, ax = plot_embedding(
        projection.sample_coords,
        _clean_labels(metadata, task),
        task["target"],
        task.get("legend_type", "discrete"),
        local,
        title,
        task.get("color_map"),
    )
    _overlay_archetypes(
        ax,
        projection.supplemental_coords,
        id_map,
        order,
        cfg,
        projection.sample_coords.columns[0],
        projection.sample_coords.columns[1],
    )
    fig.tight_layout()
    _add_archetype_marker_legend(fig, cfg, order)
    return fig


def plot_projected_pcoa(
    projection: ProjectedEmbedding,
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
    cfg: Mapping[str, Any],
    id_map: Mapping[str, str],
    order: Sequence[str],
    title: str,
):
    pc_combos = [
        tuple(x)
        for x in cfg.get("pc_combinations", [["PC1", "PC2"], ["PC2", "PC3"], ["PC1", "PC3"]])
    ]
    fig, axes = PCoAPlotProxy.plot(
        projection.sample_coords,
        projection.extra,
        _clean_labels(metadata, task),
        task,
        cfg,
        pc_combos,
    )
    for ax, (x_pc, y_pc) in zip(axes, pc_combos):
        if x_pc in projection.supplemental_coords.columns and y_pc in projection.supplemental_coords.columns:
            _overlay_archetypes(
                ax,
                projection.supplemental_coords,
                id_map,
                order,
                cfg,
                x_pc,
                y_pc,
            )
    fig.suptitle(title)
    fig.tight_layout()
    _add_archetype_marker_legend(fig, cfg, order)
    return fig


def _draw_pcoa_summary_axes(
    axes: Sequence[Any],
    projection: ProjectedEmbedding,
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
    cfg: Mapping[str, Any],
    id_map: Mapping[str, str],
    order: Sequence[str],
    pc_combos: Sequence[tuple[str, str]],
):
    coords = projection.sample_coords
    metadata = _clean_labels(metadata, task)
    common = coords.index.intersection(metadata.index, sort=False)
    coords = coords.loc[common]
    labels = metadata.loc[common, task["target"]]
    valid = labels.notna()
    coords = coords.loc[valid]
    labels = labels.loc[valid]
    legend_type = task.get("legend_type", "discrete")
    cmap_name = cfg.get("cmap", "tab20" if legend_type == "discrete" else "viridis")

    label_to_color: dict[str, Any] = {}
    if legend_type == "discrete":
        categories = sorted(labels.astype(str).unique(), key=natural_key)
        cmap = plt.get_cmap(cmap_name, max(1, len(categories)))
        color_map = task.get("color_map") or {}
        label_to_color = {
            label: color_map.get(label, cmap(i)) for i, label in enumerate(categories)
        }
        point_colors = labels.astype(str).map(label_to_color)
    else:
        numeric = pd.to_numeric(labels, errors="coerce")

    for ax, (x_pc, y_pc) in zip(axes, pc_combos):
        if legend_type == "discrete":
            ax.scatter(
                coords[x_pc],
                coords[y_pc],
                c=list(point_colors),
                alpha=float(cfg.get("alpha", 0.65)),
                s=float(cfg.get("s", 14)),
            )
        else:
            finite = numeric.notna()
            scatter = ax.scatter(
                coords.loc[finite, x_pc],
                coords.loc[finite, y_pc],
                c=numeric.loc[finite],
                cmap=cmap_name,
                alpha=float(cfg.get("alpha", 0.65)),
                s=float(cfg.get("s", 14)),
            )
            cbar = ax.figure.colorbar(scatter, ax=ax)
            cbar.set_label(task.get("colorbar_title", task["target"]))

        _overlay_archetypes(
            ax,
            projection.supplemental_coords,
            id_map,
            order,
            cfg,
            x_pc,
            y_pc,
        )
        x_explained = projection.extra.proportion_explained.iloc[int(x_pc[-1]) - 1]
        y_explained = projection.extra.proportion_explained.iloc[int(y_pc[-1]) - 1]
        ax.set_xlabel(f"{x_pc} ({x_explained:.2%})")
        ax.set_ylabel(f"{y_pc} ({y_explained:.2%})")
        ax.set_title(f"{x_pc} vs {y_pc}")
        ax.grid(True, alpha=0.15)

    if legend_type == "discrete" and axes:
        handles = [Patch(color=color, label=label) for label, color in label_to_color.items()]
        axes[-1].legend(
            handles=handles,
            title=task["target"],
            loc=cfg.get("legend_loc", "best"),
            fontsize=cfg.get("legend_fontsize", 8),
        )


def plot_k_summary(
    result: Mapping[str, Any],
    metadata: pd.DataFrame,
    task: Mapping[str, Any],
    visualization_cfg: Mapping[str, Any],
    transform: Mapping[str, Any],
    pcoa_projection: ProjectedEmbedding,
    id_map: Mapping[str, str],
    order: Sequence[str],
):
    """Consolidated per-K panel similar to the publication/final-plots layout."""
    summary_cfg = visualization_cfg.get("k_summary", {}) or {}
    simplex_cfg = visualization_cfg.get("simplex", {}) or {}
    stacked_cfg = dict(visualization_cfg.get("stacked_bar", {}) or {})
    stacked_cfg.setdefault("archetype_colors", visualization_cfg.get("archetype_colors"))
    taxa_cfg = visualization_cfg.get("taxa", {}) or {}
    pcoa_cfg = dict(visualization_cfg.get("pcoa", {}) or {})
    pcoa_cfg.setdefault("archetype_colors", visualization_cfg.get("archetype_colors"))
    pcoa_cfg.setdefault("archetype_projection", visualization_cfg.get("archetype_projection", {}))

    fig = plt.figure(figsize=tuple(summary_cfg.get("figsize", [18, 12])))
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=summary_cfg.get("width_ratios", [1.0, 1.4]),
        height_ratios=summary_cfg.get("height_ratios", [1.15, 1.0]),
    )

    ax_simplex = fig.add_subplot(outer[0, 0])
    _draw_simplex(ax_simplex, result, metadata, task, simplex_cfg, order)
    ax_simplex.set_title(summary_cfg.get("simplex_title", "Archetype Simplex"))

    ax_stacked = fig.add_subplot(outer[0, 1])
    _draw_similarity_stacked(ax_stacked, result, metadata, task, stacked_cfg, order)
    ax_stacked.set_title(summary_cfg.get("stacked_title", "Similarity Degree Stacked Bar"))

    pcoa_grid = outer[1, 0].subgridspec(1, 2)
    pcoa_axes = [fig.add_subplot(pcoa_grid[0, 0]), fig.add_subplot(pcoa_grid[0, 1])]
    pc_combos = [
        tuple(x)
        for x in summary_cfg.get(
            "pcoa_combinations",
            pcoa_cfg.get("pc_combinations", [["PC1", "PC2"], ["PC2", "PC3"]]),
        )
    ][:2]
    _draw_pcoa_summary_axes(
        pcoa_axes,
        pcoa_projection,
        metadata,
        task,
        pcoa_cfg,
        id_map,
        order,
        pc_combos,
    )

    ax_taxa = fig.add_subplot(outer[1, 1])
    _draw_taxa_stacked(ax_taxa, result, transform, taxa_cfg, order)
    ax_taxa.set_title(summary_cfg.get("taxa_title", "Taxa Contribution"))

    title = summary_cfg.get("title") or f"Archetypal analysis summary (K={result['k']})"
    fig.suptitle(title, fontsize=float(summary_cfg.get("title_fontsize", 16)))
    fig.tight_layout()
    return fig
