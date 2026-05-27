
import logging
import numpy as np
import pandas as pd
from matchms.SpectraCollection import SpectraCollection


logger = logging.getLogger("matchms")


def as_string_or_none(value):
    """Return a safe scalar string-or-None value for metadata validators."""
    if value is None or value is pd.NA:
        return None

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None

    if isinstance(value, str):
        return value

    return str(value)


def derive_metadata_column_from_column(
    collection: SpectraCollection,
    *,
    source_key: str,
    target_key: str,
    is_valid_source,
    is_valid_target,
    converter,
    clone: bool | None = True,
    source_label: str,
    target_label: str,
) -> SpectraCollection:
    """Derive one metadata column from another metadata column.

    Parameters
    ----------
    collection
        Input SpectraCollection.
    source_key
        Metadata column used as source, for example ``"smiles"``.
    target_key
        Metadata column to create or update, for example ``"inchi"``.
    is_valid_source
        Function that checks whether the source value can be converted.
    is_valid_target
        Function that checks whether the target value is already valid.
    converter
        Function converting source values to target values.
    clone
        If True, return a modified copy. If False, modify the collection in place.
    source_label
        Human-readable source name used in log messages.
    target_label
        Human-readable target name used in log messages.
    """
    target = collection.copy() if clone else collection
    metadata = target._metadata.copy()

    if source_key not in metadata.columns:
        return target

    if target_key not in metadata.columns:
        metadata[target_key] = None

    # Needed for pandas >= 3 / stricter dtype rules:
    # a column containing only np.nan may be float64 and cannot accept strings.
    metadata[target_key] = metadata[target_key].astype("object")

    def derive_value(row):
        source_value = as_string_or_none(row.get(source_key))
        target_value = as_string_or_none(row.get(target_key))

        if is_valid_target(target_value) or not is_valid_source(source_value):
            return row.get(target_key)

        converted_value = converter(source_value)

        if converted_value:
            converted_value = converted_value.rstrip()
            logger.info(
                "Added %s %s to metadata (was converted from %s)",
                target_label,
                converted_value,
                source_label,
            )
            return converted_value

        logger.warning(
            "Could not convert %s %s to %s.",
            source_label,
            source_value,
            target_label,
        )
        return row.get(target_key)

    metadata[target_key] = metadata.apply(derive_value, axis=1)

    target._metadata = metadata
    target._clear_cache(["metadata_hashes", "spectra_hashes"])

    return target
