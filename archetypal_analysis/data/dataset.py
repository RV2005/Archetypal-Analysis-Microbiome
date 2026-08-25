"""Unified dataset contract: one OTU table plus one metadata table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import pandas as pd


@dataclass
class DatasetBundle:
    otu: pd.DataFrame
    metadata: pd.DataFrame

    def aligned(self) -> "DatasetBundle":
        """Return sample-aligned tables while avoiding copies when already aligned."""
        if self.otu.index.equals(self.metadata.index):
            return self
        common_mask = self.otu.index.isin(self.metadata.index)
        otu = self.otu if bool(common_mask.all()) else self.otu.loc[common_mask]
        return DatasetBundle(otu, self.metadata.reindex(otu.index))


class UnifiedDatasetLoader:
    """Load any standardized dataset without dataset-specific code paths.

    Contract
    --------
    ``otu.tsv`` (or CSV): rows are samples, columns are taxa/features.
    ``metadata.tsv`` (or CSV): rows are samples, columns are arbitrary metadata.
    Both files use the same sample identifier as their index.

    Performance
    -----------
    Reading and sample alignment are separate from validation. When
    ``dataset.validation.enabled`` is false, the exact-match path performs only
    the two pandas reads, constant/Index-sized bookkeeping, and an ``Index.equals``
    check. It does *not* scan every OTU value, inspect every OTU dtype, coerce
    feature columns, or materialize a second OTU-sized DataFrame.

    Validation is granular. In particular, numeric validation is dtype-only by
    default. Expensive value-wise coercion of object columns is available through
    ``validation.coerce_numeric_otu`` but is deliberately opt-in because it can
    amount to reparsing the entire OTU after ``read_csv``.

    ``last_timings`` records individual phases so large-dataset runs can distinguish
    parser/I/O time from validation and alignment time.
    """

    _VALIDATION_DEFAULTS = {
        "unique_sample_ids": True,
        "numeric_otu": True,
        "coerce_numeric_otu": False,
        "sample_match": True,
        "target_column": True,
    }

    def __init__(self) -> None:
        self.last_timings: dict[str, float] = {}
        self.last_reader_info: dict[str, Any] = {}
        self.last_validation_info: dict[str, bool] = {}

    @staticmethod
    def _separator(path: Path) -> str:
        suffixes = {suffix.lower() for suffix in path.suffixes}
        return "\t" if suffixes.intersection({".tsv", ".txt"}) else ","

    @classmethod
    def _read_table(
        cls,
        path: Path,
        sample_id_column: str | None = "sample",
        *,
        low_memory: bool = True,
        memory_map: bool = False,
        retry_without_low_memory: bool = False,
    ) -> pd.DataFrame:
        """Read once with pandas, then promote the sample column to the index.

        Keeping ``index_col`` out of ``read_csv`` is intentional. In newer pandas
        versions the low-memory chunk-concatenation warning path can index a
        shortened column-name list after an index column has been removed, causing
        an internal ``IndexError`` on wide mixed-type files. Reading the sample ID
        as a normal column keeps pandas on the same fast path as a plain
        ``pd.read_csv`` call; ``set_index(..., inplace=True)`` then removes only that
        one column without materializing a second full table.
        """
        sep = cls._separator(path)
        read_kwargs = {
            "sep": sep,
            "low_memory": low_memory,
            "memory_map": memory_map,
        }
        try:
            frame = pd.read_csv(path, **read_kwargs)
        except IndexError:
            if not (low_memory and retry_without_low_memory):
                raise
            read_kwargs["low_memory"] = False
            frame = pd.read_csv(path, **read_kwargs)

        if frame.shape[1] == 0:
            raise ValueError(f"Input table has no columns: {path}")
        index_col = (
            sample_id_column
            if sample_id_column and sample_id_column in frame.columns
            else frame.columns[0]
        )
        frame.set_index(index_col, inplace=True)
        return frame

    @staticmethod
    def _normalize_index(index: pd.Index) -> pd.Index:
        if index.dtype == object and all(isinstance(value, str) for value in index[: min(len(index), 64)]):
            return index
        return index.map(str)

    @classmethod
    def _validation_options(cls, cfg: Mapping[str, Any]) -> dict[str, bool]:
        raw = cfg.get("validation", {}) or {}
        if isinstance(raw, bool):
            raw = {"enabled": raw}
        enabled = bool(raw.get("enabled", True))
        options = {
            key: bool(raw.get(key, default if enabled else False))
            for key, default in cls._VALIDATION_DEFAULTS.items()
        }
        options["enabled"] = enabled
        return options

    def load(self, otu_path: Path, metadata_path: Path, cfg: Mapping[str, Any]) -> DatasetBundle:
        total_start = perf_counter()
        timings: dict[str, float] = {}

        sample_col = cfg.get("sample_id_column", "sample")
        reader_cfg = cfg.get("reader", {}) or {}
        legacy_low_memory = bool(reader_cfg.get("low_memory", True))
        legacy_memory_map = bool(reader_cfg.get("memory_map", False))
        otu_reader_cfg = reader_cfg.get("otu", {}) or {}
        metadata_reader_cfg = reader_cfg.get("metadata", {}) or {}
        otu_low_memory = bool(
            otu_reader_cfg.get("low_memory", reader_cfg.get("otu_low_memory", legacy_low_memory))
        )
        metadata_low_memory = bool(
            metadata_reader_cfg.get("low_memory", reader_cfg.get("metadata_low_memory", legacy_low_memory))
        )
        otu_memory_map = bool(
            otu_reader_cfg.get("memory_map", reader_cfg.get("otu_memory_map", legacy_memory_map))
        )
        metadata_memory_map = bool(
            metadata_reader_cfg.get("memory_map", reader_cfg.get("metadata_memory_map", legacy_memory_map))
        )
        validation = self._validation_options(cfg)
        self.last_validation_info = validation.copy()

        start = perf_counter()
        otu = self._read_table(
            Path(otu_path),
            sample_col,
            low_memory=otu_low_memory,
            memory_map=otu_memory_map,
            retry_without_low_memory=False,
        )
        timings["read_otu_seconds"] = perf_counter() - start

        start = perf_counter()
        metadata = self._read_table(
            Path(metadata_path),
            sample_col,
            low_memory=metadata_low_memory,
            memory_map=metadata_memory_map,
            retry_without_low_memory=True,
        )
        timings["read_metadata_seconds"] = perf_counter() - start
        self.last_reader_info = {
            "otu_low_memory": otu_low_memory,
            "metadata_low_memory": metadata_low_memory,
            "otu_memory_map": otu_memory_map,
            "metadata_memory_map": metadata_memory_map,
            "index_after_read": True,
        }

        # Standardized files should already have compatible sample identifiers.
        # Index normalization is cheap relative to OTU-wide scans, but even this is
        # skipped when validation is fully disabled so the fast path stays as close
        # as possible to the two raw pd.read_csv calls.
        start = perf_counter()
        if validation["enabled"]:
            otu.index = self._normalize_index(otu.index)
            metadata.index = self._normalize_index(metadata.index)
        otu.index.name = "sample"
        metadata.index.name = "sample"
        timings["index_normalization_seconds"] = perf_counter() - start

        # Do not scan all OTU columns for cosmetic cleanup when validation is off.
        # Standardized files produced by the conversion notebook should not contain
        # Unnamed columns in the first place.
        start = perf_counter()
        if validation["enabled"]:
            unnamed = [column for column in otu.columns if str(column).startswith("Unnamed")]
            if unnamed:
                otu.drop(columns=unnamed, inplace=True)
        timings["column_cleanup_seconds"] = perf_counter() - start

        start = perf_counter()
        if validation["unique_sample_ids"]:
            if not otu.index.is_unique:
                dupes = otu.index[otu.index.duplicated()].unique().tolist()[:10]
                raise ValueError(f"OTU sample IDs must be unique; duplicates include {dupes}")
            if not metadata.index.is_unique:
                dupes = metadata.index[metadata.index.duplicated()].unique().tolist()[:10]
                raise ValueError(f"Metadata sample IDs must be unique; duplicates include {dupes}")
        timings["unique_id_validation_seconds"] = perf_counter() - start

        start = perf_counter()
        non_numeric: list[Any] = []
        if validation["numeric_otu"]:
            # This inspects the BlockManager dtype metadata only; it does not scan
            # every cell. The old implementation then reparsed every object column,
            # which can dominate runtime on large OTU matrices.
            non_numeric = [
                column
                for column, dtype in otu.dtypes.items()
                if not pd.api.types.is_numeric_dtype(dtype)
            ]
        timings["numeric_dtype_validation_seconds"] = perf_counter() - start

        start = perf_counter()
        if non_numeric:
            if validation["coerce_numeric_otu"]:
                try:
                    # Explicitly opt-in because this is O(number of OTU cells in
                    # non-numeric columns) and can effectively parse the data twice.
                    for column in non_numeric:
                        otu[column] = pd.to_numeric(otu[column], errors="raise")
                except Exception as exc:
                    raise ValueError(
                        f"OTU contains non-numeric feature columns: {non_numeric[:10]}"
                    ) from exc
            else:
                raise ValueError(
                    "OTU contains non-numeric feature columns: "
                    f"{non_numeric[:10]}. Set dataset.validation.coerce_numeric_otu=true "
                    "to coerce them, or disable numeric_otu validation for trusted standardized data."
                )
        timings["numeric_coercion_seconds"] = perf_counter() - start

        start = perf_counter()
        strict = bool(cfg.get("strict_sample_match", False))

        # Fast standardized path: exact same samples/order. Index.equals is O(number
        # of samples), not O(number of OTU cells), and no DataFrame is copied.
        if otu.index.equals(metadata.index):
            aligned_otu = otu
            aligned_metadata = metadata
        else:
            if validation["sample_match"] and strict:
                otu_only = otu.index.difference(metadata.index, sort=False)
                meta_only = metadata.index.difference(otu.index, sort=False)
                if len(otu_only) or len(meta_only):
                    raise ValueError(
                        f"OTU/metadata sample IDs differ: {len(otu_only)} OTU-only, "
                        f"{len(meta_only)} metadata-only"
                    )

            # Alignment is required pipeline behavior, not validation. It scales
            # with sample count only. The OTU is kept untouched when all OTU samples
            # are present and only metadata ordering/extra rows differ.
            common_mask = otu.index.isin(metadata.index)
            if validation["sample_match"] and not bool(common_mask.any()):
                raise ValueError("No overlapping sample IDs between OTU and metadata")

            aligned_otu = otu if bool(common_mask.all()) else otu.loc[common_mask]
            aligned_metadata = metadata.reindex(aligned_otu.index)

        timings["sample_alignment_seconds"] = perf_counter() - start
        timings["validation_seconds"] = (
            timings["unique_id_validation_seconds"]
            + timings["numeric_dtype_validation_seconds"]
            + timings["numeric_coercion_seconds"]
        )
        timings["post_read_seconds"] = perf_counter() - total_start - timings["read_otu_seconds"] - timings["read_metadata_seconds"]
        timings["total_seconds"] = perf_counter() - total_start
        self.last_timings = timings
        return DatasetBundle(aligned_otu, aligned_metadata)