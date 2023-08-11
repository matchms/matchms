import logging
import re
from matchms.filtering.filter_utils.load_known_adducts import \
    load_known_adduct_conversions


logger = logging.getLogger("matchms")


def clean_adduct(spectrum_in):
    """Clean adduct and make it consistent in style.
    Will transform adduct strings of type 'M+H+' to '[M+H]+'.

    Parameters
    ----------
    spectrum_in
        Matchms Spectrum object.
    """
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()
    adduct = spectrum.get("adduct")

    cleaned_adduct = _clean_adduct(adduct)
    if adduct != cleaned_adduct:
        spectrum.set("adduct", cleaned_adduct)
        logger.info("The adduct %d was set to %s", adduct, cleaned_adduct)
    return spectrum


def _clean_adduct(adduct: str) -> str:
    """Clean adduct and make it consistent in style.
    Will transform adduct strings of type 'M+H+' to '[M+H]+'.

    Parameters
    ----------
    adduct
        Input adduct string to be cleaned/edited.
    """
    def _get_adduct_charge(adduct):
        regex_charges = r"[1-3]{0,1}[+-]{1,2}$"
        match = re.search(regex_charges, adduct)
        if match:
            return match.group(0)
        return match

    def _adduct_conversion(adduct):
        """Convert adduct if conversion rule is known"""
        adduct_conversions = load_known_adduct_conversions()
        if adduct in adduct_conversions:
            return adduct_conversions[adduct]
        return adduct

    if not isinstance(adduct, str):
        return adduct

    adduct = adduct.strip().replace("\n", "").replace("*", "")
    adduct = adduct.replace("++", "2+").replace("--", "2-")
    if adduct.startswith("["):
        return _adduct_conversion(adduct)

    if adduct.endswith("]"):
        return _adduct_conversion("[" + adduct)

    adduct_core = "[" + adduct
    # Remove parts that can confuse the charge extraction
    for mol_part in ["CH2", "CH3", "NH3", "NH4", "O2"]:
        if mol_part in adduct:
            adduct = adduct.split(mol_part)[-1]
    adduct_charge = _get_adduct_charge(adduct)

    if adduct_charge is None:
        return _adduct_conversion(adduct_core + "]")

    adduct_cleaned = adduct_core[:-len(adduct_charge)] + "]" + adduct_charge
    return _adduct_conversion(adduct_cleaned)
