import logging
import re
from matchms.typing import SpectrumType
from ..filter_utils.load_known_adducts import load_known_adducts
from .clean_adduct import _clean_adduct


logger = logging.getLogger("matchms")


def derive_adduct_from_name(spectrum_in: SpectrumType,
                            remove_adduct_from_name: bool = True) -> SpectrumType:
    """Find adduct in compound name and add to metadata (if not present yet).

    Method to interpret the given compound name to find the adduct.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    remove_adduct_from_name:
        Remove found adducts from compound name if set to True. Default is True.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.get("compound_name", None) is not None:
        name = spectrum.get("compound_name")
    else:
        assert spectrum.get("name", None) in [None, ""], ("Found 'name' but not 'compound_name' in metadata",
                                                          "Apply 'add_compound_name' filter first.")
        return spectrum

    # Detect adduct in compound name
    adduct_from_name = None
    name_split = name.split(" ")
    for name_part in name_split[::-1][:2]:
        if _looks_like_adduct(name_part):
            adduct_from_name = name_part
            break

    if adduct_from_name and remove_adduct_from_name:
        name_adduct_removed = " ".join([x for x in name_split if x != adduct_from_name])
        name_adduct_removed = name_adduct_removed.strip("; ")
        spectrum.set("compound_name", name_adduct_removed)
        logger.info("Removed adduct %s from compound name.", adduct_from_name)

    # Add found adduct to metadata (if not present yet)
    if adduct_from_name and not _looks_like_adduct(spectrum.get("adduct")):
        adduct_cleaned = _clean_adduct(adduct_from_name)
        spectrum.set("adduct", adduct_cleaned)
        logger.info("Added adduct %s from the compound name to metadata.", spectrum.get('adduct'))

    return spectrum


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
