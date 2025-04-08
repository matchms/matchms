import logging
import math
from matchms.filtering.filter_utils.interpret_unknown_adduct import get_multiplier_and_mass_from_adduct


logger = logging.getLogger("matchms")


def require_matching_adduct_precursor_mz_parent_mass(spectrum, tolerance=0.1):
    """Checks if the adduct precursor mz and parent mass match within the tolerance"""
    if spectrum is None:
        return None

    adduct = spectrum.get("adduct")

    if adduct is None:
        logger.info("Spectrum is removed since adduct is None")
        return None
    if spectrum.get("parent_mass") is None:
        logger.info("Spectrum is removed since parent mass is None")
        return None
    if spectrum.get("precursor_mz") is None:
        logger.info("Spectrum is removed since precursor mz is None")
        return None
    try:
        precursor_mz = float(spectrum.get("precursor_mz"))
        parent_mass = float(spectrum.get("parent_mass"))
    except (TypeError, ValueError):
        logger.warning("precursor_mz or parent mass could not be converted to float, please run add_parent_mass and add_precursor_mz first")
        return spectrum

    multiplier, correction_mass = get_multiplier_and_mass_from_adduct(adduct)
    if multiplier is None:
        logger.info("Spectrum is removed since adduct: %s could not be parsed", adduct)
        return None
    expected_parent_mass = (precursor_mz - correction_mass) / multiplier
    if not math.isclose(parent_mass, expected_parent_mass, abs_tol=tolerance):
        logger.info(
            "Spectrum is removed because the adduct : %s and precursor_mz: %s suggest a parent mass of %s, but parent mass %s is given",
            adduct,
            precursor_mz,
            expected_parent_mass,
            parent_mass,
        )
        return None
    return spectrum
