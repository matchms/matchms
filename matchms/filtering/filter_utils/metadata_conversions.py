
import logging
from collections.abc import Mapping
import numpy as np
import pandas as pd


logger = logging.getLogger("matchms")


NO_METADATA_UPDATE = object()


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


def is_missing_metadata_value(value) -> bool:
    """Return True for scalar missing metadata values."""
    if value is None:
        return True

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False

    if isinstance(missing, bool):
        return missing

    return False


def apply_metadata_row_filter(
    metadata: pd.DataFrame,
    row_filter,
    *args,
    drop_missing_row_updates: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Apply a row-wise metadata filter and return updated columns.

    ``row_filter`` receives one metadata row as a mapping and must return a dict
    with metadata updates.

    Returning an empty dict means "no update".
    Returning {"key": None} means "explicitly set key to None".

    Parameters
    ----------
    metadata
        Metadata table subset.
    row_filter
        Function applied to each metadata row.
    drop_missing_row_updates
        If ``True``, missing values returned by ``row_filter`` are treated as
        "no update" and removed from the returned update table.
        If ``False``, missing values are kept as explicit updates.
    """
    records = []

    if metadata.shape[1] == 0:
        row_iter = ((index, {}) for index in metadata.index)
    else:
        row_iter = metadata.iterrows()

    for _, row in row_iter:
        updates = row_filter(row, *args, **kwargs)

        if updates is None:
            updates = {}

        if not isinstance(updates, Mapping):
            raise TypeError(
                f"Expected metadata row filter to return dict-like updates, "
                f"got {type(updates).__name__}."
            )

        records.append(dict(updates))

    updated_columns = sorted(
        {
            key
            for update_dict in records
            for key in update_dict.keys()
        }
    )

    if not updated_columns:
        return pd.DataFrame(index=metadata.index)

    records_with_all_columns = [
        {
            column: update_dict.get(column, NO_METADATA_UPDATE)
            for column in updated_columns
        }
        for update_dict in records
    ]

    updates_df = pd.DataFrame.from_records(
        records_with_all_columns,
        index=metadata.index,
    )

    if drop_missing_row_updates:
        updates_df = updates_df.mask(updates_df.map(lambda x: x is NO_METADATA_UPDATE))
        updates_df = updates_df.dropna(axis=0, how="all")
        updates_df = updates_df.dropna(axis=1, how="all")

    return updates_df


def _normalize_metadata_updates(updates, row_filter):
    if updates is None:
        return {}

    if not isinstance(updates, Mapping):
        raise TypeError(
            f"Expected metadata row filter {getattr(row_filter, '__name__', repr(row_filter))} "
            f"to return dict-like updates, got {type(updates).__name__}."
        )

    return dict(updates)


def apply_metadata_updates_to_spectrum(spectrum, updates: Mapping):
    """Apply metadata updates to a Spectrum."""
    for key, value in updates.items():
        spectrum.set(key, value)
    return spectrum
