"""Utilities for converting legacy datasets into the unified two-file schema."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .dataset import DatasetBundle, UnifiedDatasetLoader


def read_legacy_table(path: str | Path, index_col: int | str | None = None) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, index_col=index_col, low_memory=False)


def _sample_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"project", "srr"}.issubset(out.columns):
        out["sample"] = out["project"].astype(str) + "_" + out["srr"].astype(str)
        out = out.drop_duplicates("sample", keep="first").set_index("sample")
    else:
        out.index = out.index.map(str)
        out.index.name = "sample"
        if not out.index.is_unique:
            out = out.loc[~out.index.duplicated(keep="first")]
    return out


def merge_compendium_metadata(
    otu_index: pd.Index,
    sample_metadata: pd.DataFrame,
    projects: Optional[pd.DataFrame] = None,
    tags: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge legacy compendium sample/project/tag metadata into one wide table.

    Source precedence for duplicate metadata names follows the old extraction logic:
    project-level values override sample-level values, which override tag-level values.
    """
    otu_index = pd.Index(otu_index.map(str), name="sample")
    sample = _sample_key_frame(sample_metadata)
    merged = sample.reindex(otu_index).copy()

    if tags is not None:
        tags_local = tags.copy()
        if {"project", "srr", "tag", "value"}.issubset(tags_local.columns):
            tags_local["sample"] = tags_local["project"].astype(str) + "_" + tags_local["srr"].astype(str)
            wide = tags_local.pivot_table(index="sample", columns="tag", values="value", aggfunc="first")
        elif "sample" in tags_local.columns:
            wide = tags_local.drop_duplicates("sample", keep="first").set_index("sample")
        else:
            wide = tags_local
            wide.index = wide.index.map(str)
        wide = wide.reindex(otu_index)
        for col in wide.columns:
            if col not in merged.columns:
                merged[col] = wide[col]
            else:
                merged[col] = merged[col].combine_first(wide[col])

    if projects is not None:
        projects_local = projects.copy()
        projects_local.index = projects_local.index.map(str)
        projects_local = projects_local.loc[~projects_local.index.duplicated(keep="first")]
        project_ids = pd.Series(otu_index, index=otu_index).map(lambda x: x.split("_")[0])
        for col in projects_local.columns:
            project_values = project_ids.map(projects_local[col])
            # Project values overwrite duplicates to preserve the legacy extractor's precedence.
            if col in merged.columns:
                merged[col] = project_values.combine_first(merged[col])
            else:
                merged[col] = project_values

    merged.index = otu_index
    return merged


def write_standard_dataset(bundle: DatasetBundle, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    otu_path = output / "otu.tsv"
    metadata_path = output / "metadata.tsv"
    bundle.otu.to_csv(otu_path, sep="\t", index_label="sample")
    bundle.metadata.to_csv(metadata_path, sep="\t", index_label="sample")
    return otu_path, metadata_path


def convert_compendium_dataset(
    otu_file: str | Path,
    sample_metadata_file: str | Path,
    output_dir: str | Path,
    projects_file: str | Path | None = None,
    tags_file: str | Path | None = None,
) -> tuple[Path, Path]:
    otu_path = Path(otu_file)
    sep = "\t" if otu_path.suffix.lower() in {".tsv", ".txt"} else ","
    preview = pd.read_csv(otu_path, sep=sep, nrows=1)
    index_col = "sample" if "sample" in preview.columns else 0
    otu = pd.read_csv(otu_path, sep=sep, index_col=index_col, low_memory=False)
    otu.index = otu.index.map(str)
    otu.index.name = "sample"
    otu = otu.loc[:, ~otu.columns.astype(str).str.startswith("Unnamed")]

    sample_meta = read_legacy_table(sample_metadata_file, index_col=0)
    projects = read_legacy_table(projects_file, index_col=0) if projects_file else None
    tags = read_legacy_table(tags_file, index_col=None) if tags_file else None
    metadata = merge_compendium_metadata(otu.index, sample_meta, projects, tags)
    return write_standard_dataset(DatasetBundle(otu, metadata), output_dir)


def convert_direct_dataset(
    otu_file: str | Path,
    metadata_file: str | Path,
    output_dir: str | Path,
    sample_id_column: str = "sample",
) -> tuple[Path, Path]:
    loader = UnifiedDatasetLoader()
    bundle = loader.load(Path(otu_file), Path(metadata_file), {"sample_id_column": sample_id_column})
    return write_standard_dataset(bundle, output_dir)