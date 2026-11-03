"""Export matchms spectra collections to Parquet."""

from __future__ import annotations
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from matchms._parquet import (
    DEFAULT_INTENSITY_COLUMN,
    DEFAULT_MZ_COLUMN,
    DEFAULT_MZ_PRECISION,
    metadata_value_to_arrow_compatible,
    schema_metadata_for_collection,
)


if TYPE_CHECKING:
    from matchms.spectra_collection import SpectraCollection
    from matchms.spectrum import Spectrum


def save_as_parquet(
    spectra: SpectraCollection | Iterable[Spectrum],
    filename: str | Path,
    *,
    mz_column: str = DEFAULT_MZ_COLUMN,
    intensity_column: str = DEFAULT_INTENSITY_COLUMN,
    metadata_columns: Sequence[str] | None = None,
    compression: str | None = "zstd",
    mz_precision: float | None = None,
    **write_table_kwargs,
) -> None:
    """Save spectra to a row-oriented Parquet file.

    Each spectrum is written as one row. Spectrum metadata are ordinary Parquet
    columns, while fragment m/z and intensity values are numeric list columns.
    Matchms-specific collection settings are stored as Parquet schema metadata so
    a subsequent :func:`matchms.importing.load_from_parquet` call can restore the
    collection's m/z grid settings.

    Parameters
    ----------
    spectra
        A :class:`~matchms.spectra_collection.SpectraCollection` or an iterable
        of :class:`~matchms.spectrum.Spectrum` objects.
    filename
        Output Parquet file.
    mz_column
        Name of the fragment m/z list column. Default is ``"mz"``.
    intensity_column
        Name of the fragment intensity list column. Default is
        ``"intensities"``.
    metadata_columns
        Optional subset of metadata columns to write. By default all metadata
        columns are exported.
    compression
        Parquet compression codec. Default is ``"zstd"``. Set to ``None`` to
        disable compression.
    mz_precision
        m/z precision to use when ``spectra`` is an iterable of Spectrum objects.
        Ignored for an existing SpectraCollection, which keeps its own precision.
    **write_table_kwargs
        Additional keyword arguments forwarded to ``pyarrow.parquet.write_table``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if mz_column == intensity_column:
        raise ValueError("mz_column and intensity_column must have different names.")

    collection, empty = _as_collection(
        spectra,
        mz_precision=mz_precision,
    )

    if empty:
        metadata = pd.DataFrame()
        mz_lists = []
        intensity_lists = []
        collection_mz_precision = mz_precision or DEFAULT_MZ_PRECISION
        mz_rounding = "round"
        intensity_dtype = np.dtype(float)
    else:
        metadata = pd.DataFrame(collection.metadata).copy()
        if metadata_columns is not None:
            missing = [column for column in metadata_columns if column not in metadata.columns]
            if missing:
                raise ValueError(f"Metadata columns not found: {missing}.")
            metadata = metadata.loc[:, list(metadata_columns)].copy()

        collisions = [
            column
            for column in (mz_column, intensity_column)
            if column in metadata.columns
        ]
        if collisions:
            raise ValueError(
                f"Peak column name(s) {collisions} collide with metadata columns. "
                "Choose different mz_column/intensity_column names."
            )

        metadata = _normalize_metadata_dataframe(metadata)
        mz_lists = []
        intensity_lists = []
        for row_idx in range(len(collection)):
            mz, intensities = collection.fragments.get_row(row_idx)
            mz_lists.append(np.asarray(mz, dtype=float).tolist())
            intensity_lists.append(np.asarray(intensities).tolist())

        collection_mz_precision = collection.mz_precision
        mz_rounding = getattr(collection.fragments, "mz_rounding", "round")
        intensity_dtype = getattr(collection.fragments.array, "dtype", np.dtype(float))

    mz_array = pa.array(mz_lists, type=pa.list_(pa.float64()))
    intensity_arrow_type = _arrow_float_type(intensity_dtype, pa)
    intensity_array = pa.array(
        intensity_lists,
        type=pa.list_(intensity_arrow_type),
    )

    if metadata.shape[1] == 0:
        table = pa.Table.from_arrays(
            [mz_array, intensity_array],
            names=[mz_column, intensity_column],
        )
    else:
        try:
            metadata_table = pa.Table.from_pandas(metadata, preserve_index=False)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
            bad_columns = _find_arrow_incompatible_columns(metadata, pa)
            suffix = f" Problematic column(s): {bad_columns}." if bad_columns else ""
            raise TypeError(
                "Could not convert spectrum metadata to Arrow. Metadata columns must contain "
                "Parquet-compatible scalar or homogeneous nested values." + suffix
            ) from exc

        table = metadata_table.append_column(mz_column, mz_array)
        table = table.append_column(intensity_column, intensity_array)

    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata.update(
        schema_metadata_for_collection(
            mz_column=mz_column,
            intensity_column=intensity_column,
            mz_precision=collection_mz_precision,
            mz_rounding=mz_rounding,
        )
    )
    table = table.replace_schema_metadata(schema_metadata)

    pq.write_table(
        table,
        Path(filename),
        compression=compression,
        **write_table_kwargs,
    )


def _as_collection(spectra, *, mz_precision: float | None):
    from matchms.spectra_collection import SpectraCollection
    from matchms.spectrum import Spectrum

    if isinstance(spectra, SpectraCollection):
        return spectra, False

    if isinstance(spectra, Spectrum):
        spectra_list = [spectra]
    else:
        spectra_list = [spectrum for spectrum in spectra if spectrum is not None]
    if not spectra_list:
        return None, True

    precision = DEFAULT_MZ_PRECISION if mz_precision is None else float(mz_precision)
    return SpectraCollection(spectra_list, mz_precision=precision), False


def _normalize_metadata_dataframe(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    for column in metadata.columns:
        if pd.api.types.is_object_dtype(metadata[column].dtype):
            metadata[column] = metadata[column].map(metadata_value_to_arrow_compatible)
    return metadata


def _find_arrow_incompatible_columns(metadata: pd.DataFrame, pa) -> list[str]:
    bad_columns = []
    for column in metadata.columns:
        try:
            pa.array(metadata[column], from_pandas=True)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
            bad_columns.append(str(column))
    return bad_columns


def _arrow_float_type(dtype, pa):
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float16"):
        return pa.float32()
    if dtype == np.dtype("float32"):
        return pa.float32()
    return pa.float64()
