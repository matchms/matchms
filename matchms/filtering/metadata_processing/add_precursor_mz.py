import logging
import numpy as np
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import is_missing_metadata_value
from matchms.utils import get_first_common_element


logger = logging.getLogger("matchms")


_default_key = "precursor_mz"
_accepted_keys = ["precursormz", "precursor_mass"]
_accepted_types = (float, str, int)
_accepted_missing_entries = ["", "N/A", "NA", "n/a"]


def _add_precursor_mz(metadata) -> dict:
    """Add precursor_mz to correct field and make it a float.

    For missing ``precursor_mz`` field: check if there is a ``pepmass`` entry
    instead. For strings parsed as precursor m/z, convert to float.

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
        Input object with added precursor m/z metadata, or ``None`` if the input
        was ``None``.
    """
    precursor_mz_key = get_first_common_element(
        [_default_key] + _accepted_keys,
        metadata.keys(),
    )
    precursor_mz = metadata.get(precursor_mz_key)
    precursor_mz = _convert_precursor_mz(precursor_mz)

    if isinstance(precursor_mz, (float, int)):
        return {"precursor_mz": float(precursor_mz)}

    pepmass = metadata.get("pepmass", None)
    if pepmass is not None:
        try:
            pepmass_precursor_mz = pepmass[0]
        except (TypeError, IndexError):
            pepmass_precursor_mz = None

        pepmass_precursor_mz = _convert_precursor_mz(pepmass_precursor_mz)
        if pepmass_precursor_mz is not None:
            logger.warning(
                "Added precursor_mz entry based on field 'pepmass'. "
                "Consider running 'interpret_pepmass()' filter first."
            )
            return {"precursor_mz": float(pepmass_precursor_mz)}

    logger.warning("No precursor_mz found in metadata.")
    return {"precursor_mz": None}


def _convert_precursor_mz(precursor_mz):
    """Convert precursor_mz to number if possible. Otherwise return None."""
    if is_missing_metadata_value(precursor_mz):
        return None

    if isinstance(precursor_mz, str) and precursor_mz in _accepted_missing_entries:
        return None

    if not isinstance(precursor_mz, _accepted_types):
        logger.warning("Found precursor_mz of undefined type.")
        return None

    if isinstance(precursor_mz, str):
        try:
            return float(precursor_mz.strip())
        except ValueError:
            logger.warning("%s can't be converted to float.", precursor_mz)
            return None

    return precursor_mz


def _add_precursor_mz_metadata(metadata):
    """Return full metadata dict with harmonized precursor_mz.

    Kept for compatibility with existing metadata harmonization code.
    """
    metadata = metadata.copy()
    updates = _add_precursor_mz(metadata)

    metadata["precursor_mz"] = updates["precursor_mz"]

    for key in _accepted_keys:
        metadata.pop(key, None)

    return metadata


add_precursor_mz = metadata_update_filter(_add_precursor_mz, drop_missing_updates=False)