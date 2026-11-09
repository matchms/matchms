"""Build a :class:`~matchms.SpectraCollection` from parsed records without Spectrum objects."""

from __future__ import annotations
import ast
import logging
from collections.abc import Iterable, Mapping
import numpy as np
import pandas as pd
from matchms.fragment_collection import CSRFragmentCollection
from matchms.metadata import harmonize_metadata_values
from matchms.metadata_collection import (
    harmonize_metadata_collection_columns,
    harmonize_metadata_column_name,
    harmonize_metadata_types,
)
from matchms.spectra_collection import SpectraCollection


logger = logging.getLogger("matchms")

SpectrumRecord = tuple[Mapping, np.ndarray, np.ndarray]


def build_spectra_collection(
    records: Iterable[SpectrumRecord],
    *,
    metadata_harmonization: bool = True,
    mz_precision: float = 1e-6,
    mz_rounding: str = "round",
) -> SpectraCollection:
    """Build a SpectraCollection from ``(metadata, mz, intensities)`` records.

    This is the collection-level counterpart of creating one
    :class:`~matchms.spectrum.Spectrum` per record and passing the list to
    :class:`~matchms.SpectraCollection`. It produces the same metadata table and
    the same fragment array, but harmonizes metadata keys once per distinct key
    instead of once per spectrum and never materializes ``Spectrum`` objects.

    Parameters
    ----------
    records
        Iterable of ``(metadata, mz, intensities)`` tuples, one per spectrum.
        ``metadata`` is a mapping of raw keys to values as read from the file.
    metadata_harmonization
        If True, apply the matchms metadata value harmonization (pepmass,
        precursor m/z, charge, ...) to every record, as ``Spectrum`` does. Key
        names are always harmonized.
    mz_precision, mz_rounding
        Passed on to :class:`~matchms.fragment_collection.CSRFragmentCollection`.
    """
    key_cache: dict[str, str] = {}
    metadata_records = []
    lengths = []
    mz_parts = []
    intensity_parts = []

    for metadata, mz, intensities in records:
        mz = np.asarray(mz, dtype=float)
        intensities = np.asarray(intensities, dtype=float)
        if mz.ndim != 1 or mz.shape != intensities.shape:
            raise ValueError(
                f"Spectrum {len(lengths)} has {mz.size} m/z values but {intensities.size} intensities."
            )
        metadata_records.append(_harmonize_record(metadata, key_cache, metadata_harmonization))
        lengths.append(mz.size)
        mz_parts.append(mz)
        intensity_parts.append(intensities)

    n_spectra = len(lengths)
    if n_spectra == 0:
        # Same message as SpectraCollection, which the CLI relies on for empty input.
        raise ValueError("Spectra must contain at least one Spectrum.")

    metadata_frame = pd.DataFrame.from_records(metadata_records)
    if len(metadata_frame) != n_spectra:
        metadata_frame = pd.DataFrame(index=np.arange(n_spectra))
    metadata_frame = harmonize_metadata_collection_columns(metadata_frame)
    metadata_frame = harmonize_metadata_types(metadata_frame)

    fragments = CSRFragmentCollection.from_peak_arrays(
        np.asarray(lengths, dtype=np.int64),
        np.concatenate(mz_parts),
        np.concatenate(intensity_parts),
        mz_precision=mz_precision,
        mz_rounding=mz_rounding,
    )

    return SpectraCollection._from_metadata_and_fragments(
        metadata=metadata_frame.reset_index(drop=True),
        fragments=fragments,
        mz_precision=mz_precision,
    )


def _harmonize_record(metadata: Mapping, key_cache: dict[str, str], harmonize_values: bool) -> dict:
    """Harmonize one raw metadata mapping the way ``Spectrum(metadata=...)`` does.

    Keys go through :func:`harmonize_metadata_column_name` (cached per distinct
    raw key). The result has the same key order as ``PickyDict`` produces: keys
    that were already in matchms style keep their position, renamed keys follow
    in their original order. When a renamed key collides with another key, the
    value already present wins unless it is ``None``, and a differing value is
    reported, mirroring ``PickyDict``. Values are harmonized with
    :func:`harmonize_metadata_values`.
    """
    record: dict = {}
    renamed: dict = {}
    for key, value in metadata.items():
        new_key = key_cache.get(key)
        if new_key is None:
            new_key = key_cache[key] = harmonize_metadata_column_name(str(key))
        if new_key == key:
            if key in renamed:
                # A renamed key arrived first; PickyDict keeps the entry that already
                # carries the proper name and only fills it when that one is None.
                previous = renamed.pop(key)
                record[key] = previous if value is None else value
                if previous is not None and value is not None and previous != value:
                    _warn_duplicate_key(key, key)
            else:
                record[key] = value
            continue
        holder = record if new_key in record else renamed if new_key in renamed else None
        if holder is None:
            renamed[new_key] = value
        elif holder[new_key] is None:
            holder[new_key] = value
        elif holder[new_key] != value:
            _warn_duplicate_key(key, new_key)
    record.update(renamed)

    if "peak_comments" in record:
        record["peak_comments"] = ast.literal_eval(str(record["peak_comments"]))

    if harmonize_values:
        record = harmonize_metadata_values(record)
    return record


def _warn_duplicate_key(key: str, proper_key: str) -> None:
    logger.warning(
        "Key '%s' will be interpreted as '%s'. But this entry already exists. "
        "Please use '%s' if you want to replace the entry.",
        key, proper_key, proper_key,
    )
