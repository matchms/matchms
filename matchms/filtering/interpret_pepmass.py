import logging
from ..filtering.make_charge_int import _convert_charge_to_int


logger = logging.getLogger("matchms")
_accepted_types = (float, str, int)
_accepted_missing_entries = ["", "N/A", "NA", "n/a"]


def interpret_pepmass(spectrum_in):
    """Reads pepmass field (if present) and adds values to correct field(s).

    The field "pepmass" or "PEPMASS" is often used to describe the precursor ion.
    This function will interpret the values as (mz, intensity, charge) tuple. Those
    will be splitted (if present) added to the fields "precursor_mz",
    "precursor_intensity", and "charge".
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    metadata_updated = _interpret_pepmass_metadata(spectrum.metadata)
    spectrum.metadata = metadata_updated
    return spectrum


def _interpret_pepmass_metadata(metadata):
    pepmass = metadata.get("pepmass")
    if pepmass is None:
        return metadata

    mz, intensity, charge = _get_mz_intensity_charge(pepmass)
    mz = _convert_mz_or_intensity(mz)
    intensity = _convert_mz_or_intensity(intensity)
    charge = _convert_charge_to_int(charge)

    if mz is not None:
        if metadata.get("precursor_mz") is not None and metadata.get("precursor_mz") != mz:
            logger.warning("Overwriting existing precursor_mz %s with new one: %s",
                           metadata.get("precursor_mz"), str(mz))
        metadata["precursor_mz"] = mz
        logger.info("Added precursor_mz entry based on field 'pepmass'.")

    if intensity is not None:
        if metadata.get("precursor_intensity") is not None:
            logger.warning("Overwriting existing precursor_intensity %s with new one: %s",
                           metadata.get("precursor_intensity"), str(intensity))
        metadata["precursor_intensity"] = intensity
        logger.info("Added precursor_intensity entry based on field 'pepmass'.")

    if charge is not None:
        if metadata.get("charge") is not None:
            logger.warning("Overwriting existing charge %s with new one: %s",
                           metadata.get("charge"), str(charge))
        metadata["charge"] = charge
        logger.info("Added charge entry based on field 'pepmass'.")

    return metadata


def _get_mz_intensity_charge(pepmass):
    try:
        length = len(pepmass)
        values = [None, None, None]
        for i in range(length):
            values[i] = pepmass[i]
        return values[0], values[1], values[2]
    except TypeError:
        if pepmass is not None:
            return pepmass, None, None
        return None, None, None


def _convert_mz_or_intensity(entry):
    """Convert mz or intensity to number if possible. Otherwise return None."""
    if entry is None:
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
