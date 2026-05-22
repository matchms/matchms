from collections.abc import Iterable
from matchms.filtering._dispatch import collection_filter
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType
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
    if value is None:
        return True

    try:
        return value in aliases
    except TypeError:
        return False


def _harmonize_missing_entries_spectrum(
        spectrum_in: SpectrumType,
        keys: str | Iterable[str] | None = None,
        undefined=None,
        aliases: Iterable | None = None,
        clone: bool | None = True,
    ) -> SpectrumType | None:
    """Replace aliases for missing metadata entries.

    Parameters
    ----------
    spectrum_in
        Input spectrum.
    keys
        Metadata key or keys to harmonize. If ``None``, all existing metadata
        keys are harmonized.
    undefined
        Replacement value for missing entries. Default is ``None``.
    aliases
        Values that should be interpreted as missing. If ``None``,
        ``ALIASES_FOR_NONE`` is used.
    clone
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with harmonized missing metadata entries, or ``None`` if input
        was ``None``.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if aliases is None:
        aliases = ALIASES_FOR_NONE
    aliases = set(aliases)

    keys = _normalize_keys(keys)
    if keys is None:
        keys = list(spectrum.metadata.keys())

    for key in keys:
        value = spectrum.get(key)
        if _is_missing_alias(value, aliases):
            spectrum.set(key, undefined)

    return spectrum


def _harmonize_missing_entries_collection(
        spectrum_in: SpectraCollection,
        keys: str | Iterable[str] | None = None,
        undefined=None,
        aliases: Iterable | None = None,
        clone: bool | None = True,
    ) -> SpectraCollection:
    """Replace aliases for missing metadata entries in a SpectraCollection."""
    target = spectrum_in.copy() if clone else spectrum_in

    if aliases is None:
        aliases = ALIASES_FOR_NONE
    aliases = set(aliases)

    keys = _normalize_keys(keys)
    metadata = target._metadata.copy()

    if keys is None:
        keys = list(metadata.columns)
    else:
        # Preserve Spectrum behavior: missing requested keys are created.
        for key in keys:
            if key not in metadata.columns:
                metadata[key] = undefined

    for key in keys:
        metadata[key] = metadata[key].map(
            lambda value: undefined if _is_missing_alias(value, aliases) else value
        )

    target._metadata = metadata
    target._clear_cache(["metadata_hashes", "spectra_hashes"])

    return target


harmonize_missing_entries = collection_filter(
    _harmonize_missing_entries_spectrum,
    collection_impl=_harmonize_missing_entries_collection,
)
