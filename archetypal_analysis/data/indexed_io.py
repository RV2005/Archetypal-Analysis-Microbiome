"""Lossless helpers for persisted sample-indexed pipeline tables.

Pandas' dtype inference can silently change identifiers such as ``001`` to ``1`` when a
particular persisted subset happens to contain only numeric-looking sample IDs.  A different
artifact containing the same IDs plus an alphanumeric ID is then inferred as strings, making
otherwise identical samples fail to align after a round trip through TSV.

All pipeline TSV artifacts use a physical ``sample`` column.  Read that column through a
converter so its exact textual representation is retained before promoting it to the index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_sample_indexed_tsv(
    path: str | Path,
    *,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Read a pipeline TSV while preserving sample identifiers byte-for-byte as text."""
    path = Path(path)
    frame = pd.read_csv(
        path,
        sep="\t",
        converters={"sample": lambda value: str(value)},
        low_memory=low_memory,
    )
    if "sample" not in frame.columns:
        raise ValueError(f"Expected a 'sample' column in pipeline artifact: {path}")
    if frame["sample"].duplicated().any():
        examples = frame.loc[frame["sample"].duplicated(), "sample"].astype(str).unique().tolist()[:10]
        raise ValueError(f"Duplicate sample IDs in {path}: {examples}")
    frame.set_index("sample", inplace=True)
    frame.index = pd.Index(frame.index.astype(str), name="sample")
    return frame


def string_sample_index(values: Iterable[object]) -> pd.Index:
    """Return the canonical in-memory sample-index representation used by the pipeline."""
    return pd.Index([str(value) for value in values], name="sample")


def align_derived_representation(
    frame: pd.DataFrame,
    reference_index: Iterable[object],
    *,
    artifact_name: str,
    allow_positional_repair: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Align an AA/benchmark representation to the fit-input sample index.

    New artifacts should match by exact textual ID.  Older artifacts may already have lost
    formatting (e.g. ``001`` -> ``1``) when they were read with pandas dtype inference.  AA
    similarity matrices and benchmark embeddings are guaranteed to be written in fit-input row
    order, so when their row count is identical we can safely restore the authoritative
    ``fit_input`` index by position and reuse the expensive artifact.

    Returns ``(aligned_frame, repaired)``.
    """
    reference = string_sample_index(reference_index)
    local = frame.copy()
    local.index = string_sample_index(local.index)

    if local.index.equals(reference):
        return local, False

    if reference.isin(local.index).all():
        return local.reindex(reference), False

    if allow_positional_repair and len(local) == len(reference):
        local.index = reference
        return local, True

    missing = reference.difference(local.index).tolist()[:10]
    raise ValueError(
        f"{artifact_name} cannot be aligned to fit_input: "
        f"{len(reference)} reference rows, {len(local)} artifact rows; "
        f"missing sample IDs include {missing}"
    )


def _numeric_identifier(value: object):
    """Return a collision-safe numeric key for IDs changed by pandas inference."""
    from decimal import Decimal, InvalidOperation

    text = str(value).strip()
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    return number.normalize()


def align_source_frame_to_target_index(
    source: pd.DataFrame,
    target_index: Iterable[object],
    *,
    source_name: str,
) -> tuple[pd.DataFrame, int]:
    """Align an authoritative source table to a possibly legacy/coerced target index.

    Exact textual IDs are always preferred.  For unmatched IDs only, a numeric alias is
    accepted when it maps uniquely in the source (e.g. legacy ``1`` -> original ``001``).
    Ambiguous numeric aliases are never guessed.

    The returned frame is indexed by ``target_index`` so downstream persisted AA/benchmark
    artifacts can continue to be reused without rewriting them.
    """
    target = string_sample_index(target_index)
    local = source.copy()
    local.index = string_sample_index(local.index)
    if not local.index.is_unique:
        raise ValueError(f"{source_name} sample IDs must be unique before alignment")

    exact = set(local.index)
    numeric_candidates: dict[object, list[str]] = {}
    for sample_id in local.index:
        key = _numeric_identifier(sample_id)
        if key is not None:
            numeric_candidates.setdefault(key, []).append(sample_id)

    source_ids: list[str] = []
    repaired = 0
    missing: list[str] = []
    for target_id in target:
        if target_id in exact:
            source_ids.append(target_id)
            continue
        key = _numeric_identifier(target_id)
        candidates = numeric_candidates.get(key, []) if key is not None else []
        if len(candidates) == 1:
            source_ids.append(candidates[0])
            repaired += 1
        else:
            missing.append(target_id)
            source_ids.append("")

    if missing:
        raise ValueError(
            f"{source_name} cannot be aligned to fit_input sample IDs. "
            f"{len(missing)} of {len(target)} IDs have no unique match; examples: {missing[:10]}. "
            "This usually indicates stale artifacts from a different filtered dataset rather than "
            "a prediction-model problem."
        )

    aligned = local.loc[source_ids].copy()
    aligned.index = target
    return aligned, repaired
