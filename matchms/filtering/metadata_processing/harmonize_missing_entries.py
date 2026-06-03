from collections.abc import Iterable
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import is_missing_metadata_value
from matchms.utils import ALIASES_FOR_NONE


def _normalize_keys(keys):
    if keys is None:
        return None

    if isinstance(keys, str):
        return [keys]

    if isinstance(keys, Iterable):
        return list(keys)

    raise TypeError("'keys' must be None, a string, or an iterable of strings.")


def _is_missing_alias(value, aliases) -> bool:
    if is_missing_metadata_value(value):
        return True

    try:
        return value in aliases
    except TypeError:
        return False


def _harmonize_missing_entries(
    metadata,
    keys: str | Iterable[str] | None = None,
    undefined=None,
    aliases: Iterable | None = None,
) -> dict:
    """Replace aliases for missing metadata entries.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    keys
        Metadata key or keys to harmonize. If ``None``, all existing metadata
        keys are harmonized.
    undefined
        Replacement value for missing entries. Default is ``None``.
    aliases
        Values that should be interpreted as missing. If ``None``,
        ``ALIASES_FOR_NONE`` is used.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Input object with harmonized missing metadata entries, or ``None`` if
        input was ``None``.
    """
    if aliases is None:
        aliases = ALIASES_FOR_NONE
    aliases = set(aliases)

    keys = _normalize_keys(keys)
    if keys is None:
        keys = list(metadata.keys())

    updates = {}
    for key in keys:
        value = metadata.get(key)
        if _is_missing_alias(value, aliases):
            updates[key] = undefined

    return updates


harmonize_missing_entries = metadata_update_filter(
    _harmonize_missing_entries,
    drop_missing_updates=False,
)