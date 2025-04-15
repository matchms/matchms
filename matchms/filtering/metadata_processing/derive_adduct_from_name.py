import logging
import re
from typing import List, Optional
from matchms.filtering.filter_utils.interpret_unknown_adduct import get_multiplier_and_mass_from_adduct
from matchms.Spectrum import Spectrum
from matchms.typing import SpectrumType
from ..filter_utils.load_known_adducts import load_known_adducts
from .clean_adduct import _clean_adduct


logger = logging.getLogger("matchms")


def derive_adduct_from_name(
    spectrum_in: Spectrum, remove_adduct_from_name: bool = True, clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """Find adduct in compound name and add to metadata (if not present yet).

    Method to interpret the given compound name to find the adduct.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    remove_adduct_from_name:
        Remove found adducts from compound name if set to True. Default is True.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with added adduct, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    compound_name = spectrum.get("compound_name", None)
    if compound_name is None:
        if spectrum.get("name", None) not in [None, ""]:
            logger.warning("Found 'name' but not 'compound_name' in metadataApply 'add_compound_name' filter first.")
        return spectrum
    # Detect adduct in compound name
    parts_that_look_like_adduct = []
    name_split = compound_name.split(" ")
    for name_part in name_split:
        if _looks_like_adduct(name_part):
            # Some adducts occur more than once. So they are all removed.
            parts_that_look_like_adduct.append(name_part)

    if remove_adduct_from_name and len(parts_that_look_like_adduct) > 0:
        name_adduct_removed = " ".join([x for x in name_split if x not in parts_that_look_like_adduct])
        name_adduct_removed = name_adduct_removed.strip("; ")
        spectrum.set("compound_name", name_adduct_removed)
        logger.info("Removed adduct %s from compound name.", parts_that_look_like_adduct)

    if len(parts_that_look_like_adduct) > 0 and not _looks_like_adduct(spectrum.get("adduct")):
        best_adduct = _select_best_adduct(parts_that_look_like_adduct)
        if best_adduct:
            # Add found adduct to metadata (if not present yet)
            spectrum.set("adduct", best_adduct)
            logger.info("Added adduct %s from the compound name to metadata.", spectrum.get("adduct"))

    return spectrum


def _select_best_adduct(list_of_adducts: List[str]) -> Optional[str]:
    """Selects an adduct that can actually be interpreted (complete with charge and known elements)"""
    unique_cleaned_adducts = list({_clean_adduct(adduct) for adduct in list_of_adducts})
    if len(unique_cleaned_adducts) == 1:
        return unique_cleaned_adducts[0]

    completely_correct_adduct = []
    for adduct in unique_cleaned_adducts:
        multiplier, correction_mass = get_multiplier_and_mass_from_adduct(adduct)
        # check if both multiplier and correction mass are not None
        if multiplier and correction_mass:
            completely_correct_adduct.append(adduct)
    if len(completely_correct_adduct) == 0:
        return None
    if len(completely_correct_adduct) == 1:
        return completely_correct_adduct[0]
    logger.warning((
            "Two potential adducts were found in the compound name that are both valid adducts. The first adduct is "
            "used. The adducts found are: %s",
        ),
        completely_correct_adduct,
    )
    return completely_correct_adduct[0]


def _looks_like_adduct(adduct):
    """Return True if input string has expected format of an adduct."""
    if not isinstance(adduct, str):
        return False
    # Clean adduct
    adduct = _clean_adduct(adduct)
    # Load lists of default known adducts
    known_adducts = load_known_adducts()
    if adduct in list(known_adducts["adduct"]):
        return True

    # Expect format like: "[2M-H]" or "[2M+Na]+"
    regexp1 = r"^\[(([0-4]M)|(M[0-9])|(M))((Br)|(Br81)|(Cl)|(Cl37)|(S)){0,}[+-][A-Z0-9\+\-\(\)aglire]{1,}[\]0-4+-]{1,4}"
    return re.search(regexp1, adduct) is not None
