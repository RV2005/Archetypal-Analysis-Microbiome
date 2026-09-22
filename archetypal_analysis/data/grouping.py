"""Resolve repeated-measures subject identifiers across heterogeneous metadata schemas.

Large combined infant/compendium metadata tables frequently use different subject-ID columns
for different projects (for example ``SUBJECT_ID`` in one study and ``SUBJECT_ID2`` or
``SubjectID`` in another).  A single retained metadata column can therefore be entirely null
for a valid subset even though subject identifiers exist elsewhere in the standardized source
metadata.

This module resolves one group label per sample without guessing from sample IDs.  It prefers
an explicitly configured group column, can coalesce configured fallback columns, and can
auto-discover conservative subject/participant/patient identifier aliases.  When possible,
identifiers are namespaced by project/study/cohort/location so numeric IDs reused by different
studies do not collapse into one CV group.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .indexed_io import align_source_frame_to_target_index, string_sample_index


@dataclass(frozen=True)
class GroupResolution:
    groups: pd.Series
    info: dict[str, Any]


def _separator(path: Path) -> str:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    return "\t" if suffixes.intersection({".tsv", ".txt"}) else ","


def _header(path: Path) -> list[str]:
    sep = _separator(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle, delimiter=sep))


def _normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _subject_identifier_score(column: str) -> tuple[int, str] | None:
    """Return a conservative priority score for likely *individual* identifiers.

    We deliberately exclude generic ``ID`` and sample/run/accession fields.  The goal is to
    avoid silently using a per-sample identifier, which would defeat grouped CV.
    """
    norm = _normalized_name(column)
    exact_priority = {
        "subjectid": 0,
        "subject": 1,
        "participantid": 2,
        "participant": 3,
        "individualid": 4,
        "individual": 5,
        "patientid": 6,
        "patient": 7,
        "personid": 8,
        "donorid": 9,
    }
    if norm in exact_priority:
        return exact_priority[norm], str(column)
    # Common merged-table aliases such as SUBJECT_ID2 and Infantmet_Subject_ID.
    if re.search(r"subjectid\d*$", norm):
        return 10, str(column)
    if norm.endswith("subjectid"):
        return 11, str(column)
    if re.search(r"participantid\d*$", norm):
        return 12, str(column)
    if norm.endswith("participantid"):
        return 13, str(column)
    if re.search(r"individualid\d*$", norm):
        return 14, str(column)
    if norm.endswith("individualid"):
        return 15, str(column)
    if re.search(r"patientid\d*$", norm):
        return 16, str(column)
    if norm.endswith("patientid"):
        return 17, str(column)
    return None


def discover_group_columns(columns: Iterable[str]) -> list[str]:
    scored = []
    for column in columns:
        score = _subject_identifier_score(str(column))
        if score is not None:
            scored.append(score)
    return [column for _priority, column in sorted(scored, key=lambda item: (item[0], item[1].lower()))]


def _nonempty(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return series.notna() & text.str.strip().ne("") & ~text.str.lower().isin({"nan", "none", "null", "na", "n/a"})


def _coalesce_identifier_columns(frame: pd.DataFrame, candidates: Sequence[str]) -> tuple[pd.Series, pd.Series]:
    values = pd.Series(pd.NA, index=frame.index, dtype="string")
    source = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in candidates:
        if column not in frame.columns:
            continue
        valid = values.isna() & _nonempty(frame[column])
        if bool(valid.any()):
            values.loc[valid] = frame.loc[valid, column].astype("string").str.strip()
            source.loc[valid] = str(column)
    return values, source


def _family_child_identifier(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    """Conservative composite fallback for datasets storing child number within family."""
    normalized = {_normalized_name(column): column for column in frame.columns}
    family_col = normalized.get("family") or normalized.get("familyid")
    child_col = normalized.get("child") or normalized.get("childid")
    if family_col is None or child_col is None:
        return None
    valid = _nonempty(frame[family_col]) & _nonempty(frame[child_col])
    if not bool(valid.any()):
        return None
    values = pd.Series(pd.NA, index=frame.index, dtype="string")
    source = pd.Series(pd.NA, index=frame.index, dtype="string")
    values.loc[valid] = (
        frame.loc[valid, family_col].astype("string").str.strip()
        + "::"
        + frame.loc[valid, child_col].astype("string").str.strip()
    )
    source.loc[valid] = f"{family_col}+{child_col}"
    return values, source


def _namespace_values(frame: pd.DataFrame, namespace_columns: Sequence[str]) -> tuple[pd.Series, pd.Series]:
    return _coalesce_identifier_columns(frame, namespace_columns)


def _read_metadata_columns(
    path: Path,
    sample_id_column: str,
    requested: Sequence[str],
) -> pd.DataFrame:
    available = _header(path)
    sample_col = sample_id_column if sample_id_column in available else available[0]
    wanted = {sample_col, *[column for column in requested if column in available]}
    sep = _separator(path)
    frame = pd.read_csv(
        path,
        sep=sep,
        usecols=lambda column: column in wanted,
        converters={sample_col: lambda value: str(value)},
        low_memory=False,
    )
    if sample_col not in frame.columns:
        raise ValueError(f"Metadata source {path} does not contain sample ID column {sample_col!r}")
    frame.set_index(sample_col, inplace=True)
    frame.index = string_sample_index(frame.index)
    return frame


def resolve_group_labels(
    retained_metadata: pd.DataFrame,
    target_index: Iterable[object],
    *,
    preferred_column: str,
    source_metadata_path: str | Path | None = None,
    sample_id_column: str = "sample",
    fallback_columns: Sequence[str] = (),
    auto_detect: bool = True,
    namespace_columns: Sequence[str] = ("project", "Study", "StudyID", "Cohort", "location"),
) -> GroupResolution:
    """Resolve one repeated-measures group label per target sample.

    ``retained_metadata`` is tried first.  When coverage is incomplete and a canonical source
    metadata file is available, only relevant subject/namespace columns are read from that file
    and aligned to the fit-input index.  This makes grouped CV robust to metadata pruning and to
    old cached QC artifacts without forcing AA/benchmark recomputation.
    """
    target = string_sample_index(target_index)
    local = retained_metadata.copy()
    local.index = string_sample_index(local.index)
    local, retained_repairs = align_source_frame_to_target_index(
        local, target, source_name="retained metadata"
    )

    configured = []
    for column in [preferred_column, *list(fallback_columns)]:
        if column and column not in configured:
            configured.append(str(column))

    auto_columns = discover_group_columns(local.columns) if auto_detect else []
    candidates = configured + [column for column in auto_columns if column not in configured]
    groups, group_source = _coalesce_identifier_columns(local, candidates)

    canonical_repairs = 0
    canonical_columns: list[str] = []
    canonical_used = False
    source_path = Path(source_metadata_path) if source_metadata_path is not None else None
    # When the canonical metadata is available, read the small set of grouping/namespace
    # columns even if the preferred ID is already present. This prevents subject-ID
    # collisions across studies when cached QC metadata pruned project/cohort columns.
    if source_path is not None and source_path.exists():
        available = _header(source_path)
        discovered = discover_group_columns(available) if auto_detect else []
        canonical_candidates = configured + [column for column in discovered if column not in configured]
        requested = list(dict.fromkeys([*canonical_candidates, *namespace_columns, "Family", "FamilyID", "Child", "ChildID"]))
        source = _read_metadata_columns(source_path, sample_id_column, requested)
        source, canonical_repairs = align_source_frame_to_target_index(
            source, target, source_name="standardized source metadata"
        )
        canonical_columns = source.columns.astype(str).tolist()
        canonical_used = True

        source_groups, source_group_source = _coalesce_identifier_columns(source, canonical_candidates)
        family_child = _family_child_identifier(source)
        if family_child is not None:
            family_values, family_source = family_child
            fill = source_groups.isna() & family_values.notna()
            source_groups.loc[fill] = family_values.loc[fill]
            source_group_source.loc[fill] = family_source.loc[fill]

        fill = groups.isna() & source_groups.notna()
        groups.loc[fill] = source_groups.loc[fill]
        group_source.loc[fill] = source_group_source.loc[fill]
        local_for_namespace = source
    else:
        local_for_namespace = local

    # Namespace identifiers to prevent e.g. subject '1' from two projects becoming one group.
    namespace, namespace_source = _namespace_values(local_for_namespace, namespace_columns)
    resolved = pd.Series(pd.NA, index=target, dtype="string", name="__resolved_group__")
    valid = groups.notna()
    with_namespace = valid & namespace.notna()
    without_namespace = valid & namespace.isna()
    resolved.loc[with_namespace] = (
        namespace_source.loc[with_namespace].astype("string")
        + "="
        + namespace.loc[with_namespace].astype("string")
        + "::"
        + groups.loc[with_namespace].astype("string")
    )
    resolved.loc[without_namespace] = (
        group_source.loc[without_namespace].astype("string")
        + "::"
        + groups.loc[without_namespace].astype("string")
    )

    source_counts = group_source.dropna().value_counts().to_dict()
    info = {
        "preferred_column": str(preferred_column),
        "configured_fallback_columns": list(map(str, fallback_columns)),
        "auto_detect": bool(auto_detect),
        "namespace_columns": list(map(str, namespace_columns)),
        "resolved_rows": int(resolved.notna().sum()),
        "unresolved_rows": int(resolved.isna().sum()),
        "unique_groups": int(resolved.dropna().nunique()),
        "identifier_columns_used": {str(key): int(value) for key, value in source_counts.items()},
        "canonical_metadata_used": canonical_used,
        "canonical_metadata_columns_loaded": canonical_columns,
        "retained_metadata_id_repairs": int(retained_repairs),
        "canonical_metadata_id_repairs": int(canonical_repairs),
    }
    return GroupResolution(groups=resolved, info=info)
