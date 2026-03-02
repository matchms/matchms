import logging
import re
from typing import Optional
from matchms.filtering.filter_utils.interpret_unknown_adduct import get_charge_of_adduct
from matchms.filtering.filter_utils.load_known_adducts import load_known_adduct_conversions
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def clean_adduct(spectrum_in, clone: Optional[bool] = True) -> Optional[SpectrumType]:
    """Clean adduct and make it consistent in style.
    Will transform adduct strings of type 'M+H+' to '[M+H]+'.

    Parameters
    ----------
    spectrum_in:
        Matchms Spectrum object.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with cleaned adduct, or `None` if not present.
    """
    if spectrum_in is None:
        return None
    adduct = spectrum_in.get("adduct")
    if adduct is None:
        return spectrum_in

    spectrum = spectrum_in.clone() if clone else spectrum_in

    cleaned_adduct = _clean_adduct(adduct, spectrum.get("charge"))

    if spectrum.get("charge"):
        if spectrum.get("charge") != get_charge_of_adduct(cleaned_adduct):
            logger.warning(
                "The charge in the adduct: %s and the given charge: %s do not match", adduct, spectrum.get("charge")
            )
    else:
        # set charge to adduct
        charge_from_adduct = get_charge_of_adduct(cleaned_adduct)
        if charge_from_adduct:
            logger.info(
                "Unknown charge was derived from adduct: %s, now charge is %s", cleaned_adduct, charge_from_adduct
            )
            spectrum.set("charge", charge_from_adduct)

    if adduct != cleaned_adduct:
        spectrum.set("adduct", cleaned_adduct)
        logger.info("The adduct %s was set to %s", adduct, cleaned_adduct)
    return spectrum


def _clean_adduct(adduct: str, charge=None) -> str:
    """Clean adduct and make it consistent in style.
    Will transform adduct strings of type 'M+H+' to '[M+H]+'.

    Parameters
    ----------
    adduct
        Input adduct string to be cleaned/edited.
    """
    if not isinstance(adduct, str):
        return adduct

    adduct = adduct.strip().replace("\n", "").replace("*", "")
    adduct = adduct.replace("++", "2+").replace("--", "2-")

    if not adduct.startswith("["):
        adduct = _add_missing_brackets_to_adduct(adduct)
    if adduct.endswith("]"):
        charge = _convert_int_charge_to_str(charge)
        if charge:
            adduct += charge
    return _convert_known_adduct(adduct)


def _add_missing_brackets_to_adduct(adduct):
    """Adds missing brackets to an adduct and moves the charge outside."""

    def _get_adduct_charge(adduct):
        # Remove parts that can confuse the charge extraction. Because they end with a number and the ] is missing.
        for mol_part in ["CH2", "CH3", "NH3", "NH4", "O2"]:
            if mol_part in adduct:
                adduct = adduct.split(mol_part)[-1]
        regex_charges = r"[1-3]{0,1}[+-]{1,2}$"
        match = re.search(regex_charges, adduct)
        if match:
            return match.group(0)
        return match

    if not adduct.startswith("["):
        adduct = "[" + adduct
    if adduct.endswith("]"):
        return adduct

    adduct_charge = _get_adduct_charge(adduct)

    if adduct_charge is None:
        return adduct + "]"
    return adduct[: -len(adduct_charge)] + "]" + adduct_charge


def _convert_int_charge_to_str(charge):
    """Converts integer to str format of charge

    e.g.:
    1 -> +
    -1 -> -
    2 -> 2+
    -2 -> 2-
    """
    if charge is None:
        return None
    if not isinstance(charge, int):
        logger.warning("Charge is not given as int. Apply 'make_charge_int' filter first.")
        return None
    if charge == 0:
        return None
    if charge < 0:
        sign = "-"
    else:
        sign = "+"
    if charge in (-1, 1):
        return sign
    return str(abs(charge)) + sign


def _convert_known_adduct(adduct):
    """Convert adduct if conversion rule is known"""
    adduct_conversions = load_known_adduct_conversions()
    if adduct in adduct_conversions:
        return adduct_conversions[adduct]
    return adduct
