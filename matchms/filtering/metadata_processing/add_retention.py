import logging
import re
from typing import Any
from matchms.filtering._dispatch import metadata_update_filter
from matchms.utils import filter_none, get_common_keys


logger = logging.getLogger("matchms")


_retention_time_keys = [
    "retention_time",
    "retentiontime",
    "rt",
    "scan_start_time",
    "rt_query",
    "rtinseconds",
]
_retention_index_keys = ["retention_index", "retentionindex", "ri"]


def _add_retention_time(metadata) -> dict:
    """Add retention time information to the ``retention_time`` key as float.

    Negative values and values that cannot be converted to float result in no
    update for ``retention_time``.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Input object with harmonized retention time metadata, or ``None`` if the
        input was ``None``.
    """
    return _get_retention_update(
        metadata,
        target_key="retention_time",
        accepted_keys=_retention_time_keys,
    )


def _add_retention_index(metadata) -> dict:
    """Add retention index information to the ``retention_index`` key as float.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Input object with harmonized retention index metadata, or ``None`` if
        the input was ``None``.
    """
    return _get_retention_update(
        metadata,
        target_key="retention_index",
        accepted_keys=_retention_index_keys,
    )


def _get_retention_update(metadata, target_key: str, accepted_keys: list[str]) -> dict:
    """Return metadata update for a retention target key."""
    common_keys = get_common_keys(metadata.keys(), accepted_keys)

    if len(common_keys) == 0:
        return {}

    values_for_keys = filter_none([metadata[key] for key in common_keys])
    values = list(map(_safe_convert_to_float, values_for_keys))
    value = next(filter_none(values), None)

    return {target_key: value}


def _safe_store_value(metadata: dict, value: Any, target_key: str) -> dict:
    """Safely store a value under target_key.

    Kept for compatibility with existing metadata harmonization code.
    """
    if value is not None:
        value = _safe_convert_to_float(value)
    metadata[target_key] = value
    return metadata


def _safe_convert_to_float(retention_time: Any) -> float | None:
    """Safely convert value to float. Return None on failure."""
    if isinstance(retention_time, list):
        if len(retention_time) == 1:
            retention_time = retention_time[0]
        else:
            return None

    if isinstance(retention_time, str):
        retention_time = retention_time.strip().replace(",", ".")
        pattern = r"^([+-]?\d*\.?\d+)\s*(min|s|h|ms|sec)$"
        conversion = {"min": 60, "s": 1, "h": 3600, "ms": 1e-3, "sec": 1}
        match = re.search(pattern, retention_time)

        if match and len(match.groups()) == 2:
            value = match.group(1)
            unit = match.group(2)
            return float(value) * conversion[unit]

    try:
        retention_time = float(retention_time)
        return retention_time if retention_time >= 0 else None
    except (ValueError, TypeError):
        logger.warning("%s can't be converted to float.", str(retention_time))
        return None


def _add_retention(metadata: dict, target_key: str, accepted_keys: list[str]) -> dict:
    """Add value from one of accepted keys to target key.

    To be used for existing metadata harmonization code.
    This function returns the full metadata dictionary.
    """
    updates = _get_retention_update(metadata, target_key, accepted_keys)

    if target_key in updates:
        metadata[target_key] = updates[target_key]
    else:
        metadata[target_key] = None

    return metadata


add_retention_time = metadata_update_filter(_add_retention_time)
add_retention_index = metadata_update_filter(_add_retention_index)