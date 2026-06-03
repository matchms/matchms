import logging
import re
import numpy as np
import pandas as pd
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.metadata_conversions import is_missing_metadata_value
from matchms.typing import SpectraCollectionType, SpectrumType
from .make_charge_int import _convert_charge_to_int


logger = logging.getLogger("matchms")

_accepted_types = (float, str, int)
_accepted_missing_entries = ["", "N/A", "NA", "n/a"]


def _interpret_pepmass_spectrum(
    spectrum_in,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Reads pepmass field, if present, and adds values to correct fields.

    The field ``pepmass`` or ``PEPMASS`` is often used to describe the precursor
    ion. This function interprets the values as ``(mz, intensity, charge)`` and
    stores them in ``precursor_mz``, ``precursor_intensity``, and ``charge``.

    Parameters
    ----------
    spectrum_in
        Input spectrum.
    clone
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with interpreted pepmass metadata, or ``None`` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in
    spectrum.metadata = _interpret_pepmass_metadata(spectrum.metadata)
    return spectrum


def _interpret_pepmass_collection(
    spectrum_in: SpectraCollectionType,
    clone: bool | None = True,
) -> SpectraCollectionType:
    """Reads pepmass field, if present, for a SpectraCollection."""
    target = spectrum_in.copy() if clone else spectrum_in
    metadata = target._metadata.copy()

    if "pepmass" not in metadata.columns:
        return target

    metadata = metadata.apply(
        lambda row: pd.Series(_interpret_pepmass_metadata(row.to_dict())),
        axis=1,
    )

    metadata = metadata.dropna(axis=1, how="all")

    target._metadata = metadata.reset_index(drop=True)
    target._clear_cache(["metadata_hashes", "spectra_hashes"])
    return target


def _interpret_pepmass_metadata(metadata):
    """Return full metadata with interpreted pepmass entries."""
    metadata = metadata.copy()

    pepmass = metadata.get("pepmass")
    if is_missing_metadata_value(pepmass):
        return metadata

    mz, intensity, charge = _get_mz_intensity_charge(pepmass)
    mz = _convert_mz_or_intensity(mz)
    intensity = _convert_mz_or_intensity(intensity)
    charge = _convert_charge_to_int(charge)

    if mz is not None:
        if metadata.get("precursor_mz") is not None and _substantial_difference(
            metadata.get("precursor_mz"),
            mz,
            atol=0.001,
        ):
            logger.warning(
                "Overwriting existing precursor_mz %s with new one: %s",
                metadata.get("precursor_mz"),
                str(mz),
            )
        metadata["precursor_mz"] = mz
        logger.info("Added precursor_mz entry based on field 'pepmass'.")

    if intensity is not None:
        if metadata.get("precursor_intensity") is not None:
            logger.warning(
                "Overwriting existing precursor_intensity %s with new one: %s",
                metadata.get("precursor_intensity"),
                str(intensity),
            )
        metadata["precursor_intensity"] = intensity
        logger.info("Added precursor_intensity entry based on field 'pepmass'.")

    if charge is not None:
        if metadata.get("charge") is not None:
            logger.warning(
                "Overwriting existing charge %s with new one: %s",
                metadata.get("charge"),
                str(charge),
            )
        metadata["charge"] = charge
        logger.info("Added charge entry based on field 'pepmass'.")

    metadata.pop("pepmass", None)
    logger.info("Removed pepmass, since the information was added to other fields")

    return metadata


def _get_mz_intensity_charge(pepmass):
    try:
        if isinstance(pepmass, str):
            matches = re.findall(r"\(([^)]+)\)", pepmass)
            if len(matches) > 1:
                raise ValueError("Found more than one tuple in pepmass field.")
            if len(matches) == 1:
                pepmass = matches[0].split(",")
            if len(matches) == 0:
                try:
                    pepmass = float(pepmass)
                except ValueError:
                    return None, None, None

        length = len(pepmass)
        values = [None, None, None]
        for i in range(min(length, 3)):
            values[i] = pepmass[i]

        return values[0], values[1], values[2]

    except TypeError:
        if pepmass is not None:
            return pepmass, None, None
        return None, None, None


def _convert_mz_or_intensity(entry):
    """Convert mz or intensity to number if possible. Otherwise return None."""
    if is_missing_metadata_value(entry):
        return None

    if isinstance(entry, str) and entry in _accepted_missing_entries:
        return None

    if not isinstance(entry, _accepted_types):
        logger.warning("Found undefined type.")
        return None

    if isinstance(entry, str):
        try:
            return float(entry.strip())
        except ValueError:
            logger.warning("%s can't be converted to float.", entry)
            return None

    return entry


def _substantial_difference(mz_now, mz_new, atol=0.001):
    """Return True if mz_now and mz_new differ by more than atol."""
    if is_missing_metadata_value(mz_now):
        return True

    try:
        mz_now_float = float(mz_now)
    except (ValueError, TypeError):
        return True

    return np.abs(mz_now_float - mz_new) > atol


interpret_pepmass = collection_filter(
    _interpret_pepmass_spectrum,
    collection_impl=_interpret_pepmass_collection,
)