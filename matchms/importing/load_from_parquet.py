"""Import Parquet datasets as :class:`matchms.SpectraCollection`."""

from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.sparse import coo_array, csr_array
from matchms._parquet import (
    DEFAULT_MZ_PRECISION,
    DEFAULT_MZ_ROUNDING,
    INTENSITY_COLUMN_CANDIDATES,
    MZ_COLUMN_CANDIDATES,
    decode_schema_metadata,
    normalize_requested_columns,
    resolve_peak_column,
)
from matchms.fragment_collection import CSRFragmentCollection
from matchms.metadata_collection import (
    harmonize_metadata_collection_columns,
    harmonize_metadata_types,
)
from matchms.spectra_collection import SpectraCollection


def load_from_parquet(
    filename: str | Path,
    *,
    mz_column: str | None = None,
    intensity_column: str | None = None,
    metadata_harmonization: bool = True,
    mz_precision: float | None = None,
    mz_rounding: str | None = None,
    columns: Sequence[str] | None = None,
    filters: Any = None,
    **read_table_kwargs,
) -> SpectraCollection:
    """Load spectra from a Parquet file or dataset into a SpectraCollection.

    Parquet is treated as a row-oriented spectra table: each row represents one
    spectrum, peak m/z values are stored in one list column, peak intensities in
    another list column, and all remaining columns become spectrum metadata.

    Peak columns can be given explicitly with ``mz_column`` and
    ``intensity_column``. If omitted, matchms first checks schema metadata written
    by :func:`matchms.exporting.save_as_parquet` and otherwise auto-detects common
    names, including ``ms2_mzs`` and ``ms2_normalized_intensities``.

    Parameters
    ----------
    filename
        Path to a Parquet file or partitioned Parquet dataset.
    mz_column
        Name of the list column containing fragment m/z values. If ``None``,
        infer it from matchms schema metadata or well-known column names.
    intensity_column
        Name of the list column containing fragment intensities. If ``None``,
        infer it from matchms schema metadata or well-known column names.
    metadata_harmonization
        If ``True``, harmonize metadata column names and canonical scalar dtypes
        using the same collection-level helpers as ``SpectraCollection``.
    mz_precision
        m/z grid precision used by the collection fragment backend. If omitted,
        use the value stored by matchms in the Parquet schema, or ``1e-6`` for
        third-party Parquet data.
    mz_rounding
        m/z binning mode, ``"round"`` or ``"floor"``. If omitted, use the value
        stored in the Parquet schema, or ``"round"`` for third-party data.
    columns
        Optional metadata columns to read. Peak columns are added automatically,
        so callers do not need to include them when explicit peak-column names
        are supplied. This is useful for wide Parquet datasets.
    filters
        Row filters using the syntax accepted by ``pyarrow.parquet.read_table``;
        for example ``[("ingest_lib", "!=", "enveda-180")]``.
    **read_table_kwargs
        Additional keyword arguments forwarded to ``pyarrow.parquet.read_table``.

    Returns
    -------
    SpectraCollection
        Imported spectra. Fragment arrays are constructed directly from Arrow
        list columns without creating intermediate ``Spectrum`` objects.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"The specified file or dataset: {filename} doesn't exist.")

    # Lazy import of pyarrow
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _read_parquet_schema(path, pq)
    schema_metadata = decode_schema_metadata(schema.metadata)

    resolved_mz_column = resolve_peak_column(
        schema.names,
        explicit_column=mz_column,
        schema_column=schema_metadata.get("matchms_mz_column"),
        candidates=MZ_COLUMN_CANDIDATES,
        kind="mz",
    )
    resolved_intensity_column = resolve_peak_column(
        schema.names,
        explicit_column=intensity_column,
        schema_column=schema_metadata.get("matchms_intensity_column"),
        candidates=INTENSITY_COLUMN_CANDIDATES,
        kind="intensity",
    )

    read_columns = normalize_requested_columns(
        columns,
        resolved_mz_column,
        resolved_intensity_column,
    )

    table = pq.read_table(
        path,
        columns=read_columns,
        filters=filters,
        **read_table_kwargs,
    )

    if table.num_rows == 0:
        raise ValueError("The Parquet selection contains no spectra.")

    lengths, all_mz = _flatten_list_column(
        table[resolved_mz_column],
        column_name=resolved_mz_column,
        pa=pa,
    )
    intensity_lengths, all_intensities = _flatten_list_column(
        table[resolved_intensity_column],
        column_name=resolved_intensity_column,
        pa=pa,
    )

    if not np.array_equal(lengths, intensity_lengths):
        mismatched_rows = np.flatnonzero(lengths != intensity_lengths)
        first_row = int(mismatched_rows[0])
        raise ValueError(
            "Peak m/z and intensity list lengths must match for every spectrum. "
            f"Row {first_row} has {lengths[first_row]} m/z values and "
            f"{intensity_lengths[first_row]} intensity values."
        )

    metadata = _metadata_dataframe(
        table,
        mz_column=resolved_mz_column,
        intensity_column=resolved_intensity_column,
    )
    if metadata_harmonization:
        metadata = harmonize_metadata_collection_columns(metadata)
        metadata = harmonize_metadata_types(metadata)

    resolved_mz_precision = _resolve_mz_precision(mz_precision, schema_metadata)
    resolved_mz_rounding = mz_rounding or schema_metadata.get("matchms_mz_rounding", DEFAULT_MZ_ROUNDING)

    fragments = _build_fragment_collection(
        lengths=lengths,
        mz=all_mz,
        intensities=all_intensities,
        mz_precision=resolved_mz_precision,
        mz_rounding=resolved_mz_rounding,
    )

    return SpectraCollection._from_metadata_and_fragments(
        metadata=metadata,
        fragments=fragments,
        mz_precision=resolved_mz_precision,
    )


def _read_parquet_schema(path: Path):
    """Read schema for a Parquet file or partitioned dataset."""
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    try:
        return pq.read_schema(path)
    except (OSError, ValueError):
        return ds.dataset(path, format="parquet").schema


def _flatten_list_column(column, *, column_name: str, pa) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row list lengths and one flat floating-point NumPy array."""
    array = column.combine_chunks()

    if array.null_count:
        null_rows = np.flatnonzero(np.asarray(array.is_null().to_numpy(zero_copy_only=False)))
        raise ValueError(
            f"Peak column {column_name!r} contains null lists at rows {null_rows[:5].tolist()}. "
            "Use empty lists for spectra without peaks."
        )

    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        lengths = np.diff(offsets)
        values = array.values
    elif pa.types.is_fixed_size_list(array.type):
        lengths = np.full(len(array), array.type.list_size, dtype=np.int64)
        values = array.values
    else:
        raise ValueError(
            f"Peak column {column_name!r} must be an Arrow list column, got {array.type}."
        )

    if not (
        pa.types.is_floating(values.type)
        or pa.types.is_integer(values.type)
        or pa.types.is_decimal(values.type)
    ):
        raise ValueError(
            f"Peak column {column_name!r} must contain numeric values, got {values.type}."
        )

    flat = np.asarray(values.to_numpy(zero_copy_only=False), dtype=float)
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"Peak column {column_name!r} contains non-finite values.")

    return lengths, flat


def _metadata_dataframe(table, *, mz_column: str, intensity_column: str) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in table.column_names
        if column not in (mz_column, intensity_column)
    ]
    if not metadata_columns:
        return pd.DataFrame(index=np.arange(table.num_rows))
    return table.select(metadata_columns).to_pandas().reset_index(drop=True)


def _resolve_mz_precision(
    explicit_mz_precision: float | None,
    schema_metadata: dict[str, str],
) -> float:
    if explicit_mz_precision is not None:
        return float(explicit_mz_precision)

    stored = schema_metadata.get("matchms_mz_precision")
    if stored is None:
        return DEFAULT_MZ_PRECISION

    try:
        return float(stored)
    except ValueError as exc:
        raise ValueError(f"Invalid matchms_mz_precision stored in Parquet metadata: {stored!r}.") from exc


def _build_fragment_collection(
    *,
    lengths: np.ndarray,
    mz: np.ndarray,
    intensities: np.ndarray,
    mz_precision: float,
    mz_rounding: str,
) -> CSRFragmentCollection:
    n_spectra = len(lengths)
    template = CSRFragmentCollection.from_array(
        csr_array((n_spectra, 0), dtype=float),
        mz_precision=mz_precision,
        mz_rounding=mz_rounding,
    )

    if mz.size == 0:
        return template

    if np.any(mz < 0):
        raise ValueError("Peak m/z values must be non-negative.")

    bin_idx = template.mz_to_bin(mz)
    row_idx = np.repeat(np.arange(n_spectra, dtype=np.int64), lengths)
    n_bins = int(bin_idx.max()) + 1

    fragments_array = coo_array(
        (intensities, (row_idx, bin_idx)),
        shape=(n_spectra, n_bins),
    ).tocsr()

    return template._from_array(fragments_array)
