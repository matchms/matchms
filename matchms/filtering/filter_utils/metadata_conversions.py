
import logging
from collections.abc import Mapping
import numpy as np
import pandas as pd


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


def as_float_or_none(value):
    """Return a safe scalar float-or-None value for metadata calculations."""
    if value is None or value is pd.NA:
        return None

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    raise ValueError(f"Expected scalar numeric metadata value, got {type(value).__name__}.")


def apply_metadata_row_filter(
    metadata: pd.DataFrame,
    row_filter,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Apply a row-wise metadata filter and return only updated columns.

    ``row_filter`` receives one metadata row as a mapping and must return a dict
    with metadata updates. Returning an empty dict or ``None`` means "no changes".

    The returned DataFrame has the same index as ``metadata`` and only contains
    columns that were updated or created.
    """

    def _apply_row(row):
        updates = row_filter(row, *args, **kwargs)

        if updates is None:
            return {}

        if not isinstance(updates, Mapping):
            raise TypeError(
                f"Expected metadata row filter to return dict-like updates, "
                f"got {type(updates).__name__}."
            )

        return dict(updates)


    updates = metadata.apply(_apply_row, axis=1)

    records = [
        update
        for update in updates
        if isinstance(update, Mapping) and len(update) > 0
    ]

    if not records:
        return pd.DataFrame(index=metadata.index)

    # Need to preserve the original row index only for rows that actually
    # produced updates.
    update_indices = [
        index
        for index, update in updates.items()
        if isinstance(update, Mapping) and len(update) > 0
    ]

    return pd.DataFrame.from_records(records, index=update_indices)


def apply_metadata_updates_to_spectrum(spectrum, updates: Mapping):
    """Apply metadata updates to a Spectrum."""
    for key, value in updates.items():
        spectrum.set(key, value)
    return spectrum
