import logging

from matchms.filtering.load_adducts import load_adducts_dict
from matchms.constants import PROTON_MASS
from matchms.filtering.repair_adduct.clean_adduct import clean_adduct

logger = logging.getLogger("matchms")


def derive_parent_mass_from_precursor_mz(spectrum_in, estimate_from_adduct):
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    precursor_mz = spectrum.get("precursor_mz", None)
    if precursor_mz is None:
        logger.warning("Missing precursor m/z to derive parent mass.")
        return None
    adducts_dict = load_adducts_dict()
    charge = _get_charge(spectrum)
    adduct = clean_adduct(spectrum.get("adduct"))

    if estimate_from_adduct and (adduct in adducts_dict):
        multiplier = adducts_dict[adduct]["mass_multiplier"]
        correction_mass = adducts_dict[adduct]["correction_mass"]
        if correction_mass is not None and multiplier is not None:
            parent_mass = (precursor_mz - correction_mass) / multiplier
            return parent_mass

    if _is_valid_charge(charge):
        # Assume adduct of shape [M+xH] or [M-xH]
        protons_mass = PROTON_MASS * charge
        precursor_mass = precursor_mz * abs(charge)
        parent_mass = precursor_mass - protons_mass
        return parent_mass


def derive_precursor_mz_from_parent_mass(spectrum_in):
    """Derives the precursor_mz from the parent mass and adduct or charge"""
    estimate_from_adduct = True
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    parent_mass = spectrum.get("parent_mass")
    if parent_mass is None:
        logger.warning("Missing parent mass to derive precursor mz.")
        return None
    adduct = clean_adduct(spectrum.get("adduct"))
    adducts_dict = load_adducts_dict()
    if estimate_from_adduct and (adduct in adducts_dict):
        multiplier = adducts_dict[adduct]["mass_multiplier"]
        correction_mass = adducts_dict[adduct]["correction_mass"]
        if correction_mass is not None and multiplier is not None:
            precursor_mz = parent_mass * multiplier + correction_mass
            return precursor_mz

    charge = _get_charge(spectrum)
    if _is_valid_charge(charge):
        # Assume adduct of shape [M+xH] or [M-xH]
        protons_mass = PROTON_MASS * charge
        precursor_mass = parent_mass + protons_mass
        precursor_mz = precursor_mass / abs(charge)
        return precursor_mz


def _is_valid_charge(charge):
    return (charge is not None) and (charge != 0)


def _get_charge(spectrum):
    """Get charge from `Spectrum()` object.
    In case no valid charge is found, guess +1 or -1 based on ionmode.
    Else return 0.
    """
    charge = spectrum.get("charge")
    if _is_valid_charge(charge):
        return charge
    if spectrum.get('ionmode') == "positive":
        logger.info(
            "Missing charge entry, but positive ionmode detected. "
            "Consider prior run of `correct_charge()` filter.")
        return 1
    if spectrum.get('ionmode') == "negative":
        logger.info(
            "Missing charge entry, but negative ionmode detected. "
            "Consider prior run of `correct_charge()` filter.")
        return -1

    logger.warning(
        "Missing charge and ionmode entries. "
        "Consider prior run of `derive_ionmode()` and `correct_charge()` filters.")
    return 0
