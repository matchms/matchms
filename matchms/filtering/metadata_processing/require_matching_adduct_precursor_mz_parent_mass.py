import logging
import math
from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.interpret_unknown_adduct import (
    get_multiplier_and_mass_from_adduct,
)
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
    is_missing_metadata_value,
)


logger = logging.getLogger("matchms")


def _safe_float_or_none(value):
    """Return float or None without raising for invalid metadata values."""
    try:
        return as_float_or_none(value)
    except ValueError:
        return None


def _require_matching_adduct_precursor_mz_parent_mass(
    metadata,
    tolerance=0.1,
) -> bool:
    """Check if adduct, precursor m/z, and parent mass match within tolerance.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    tolerance
        Absolute tolerance used to compare the given parent mass to the parent
        mass implied by ``precursor_mz`` and ``adduct``. Default is ``0.1``.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Spectrum input is returned unchanged if adduct, precursor m/z, and parent
        mass match, otherwise ``None``. SpectraCollection input is returned with
        non-matching rows removed.
    """
    adduct = as_string_or_none(metadata.get("adduct"))

    if adduct is None:
        logger.info("Spectrum is removed since adduct is None")
        return False

    if is_missing_metadata_value(metadata.get("parent_mass")):
        logger.info("Spectrum is removed since parent mass is None")
        return False

    if is_missing_metadata_value(metadata.get("precursor_mz")):
        logger.info("Spectrum is removed since precursor mz is None")
        return False

    precursor_mz = _safe_float_or_none(metadata.get("precursor_mz"))
    parent_mass = _safe_float_or_none(metadata.get("parent_mass"))

    if precursor_mz is None or parent_mass is None:
        logger.warning(
            "precursor_mz or parent mass could not be converted to float, please run add_parent_mass and"
            "add_precursor_mz first"
        )
        return False

    multiplier, correction_mass = get_multiplier_and_mass_from_adduct(adduct)
    if multiplier is None:
        logger.info("Spectrum is removed since adduct: %s could not be parsed", adduct)
        return False

    expected_parent_mass = (precursor_mz - correction_mass) / multiplier

    if not math.isclose(parent_mass, expected_parent_mass, abs_tol=tolerance):
        logger.info(
            "Spectrum is removed because the adduct : %s and precursor_mz: %s suggest a parent mass of %s, but"
            " parent mass %s is given",
            adduct,
            precursor_mz,
            expected_parent_mass,
            parent_mass,
        )
        return False

    return True


require_matching_adduct_precursor_mz_parent_mass = metadata_requirement_filter(
    _require_matching_adduct_precursor_mz_parent_mass
)