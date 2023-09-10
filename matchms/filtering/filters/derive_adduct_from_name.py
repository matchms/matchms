import logging
import re
from matchms.typing import SpectrumType
from ..filter_utils.load_known_adducts import load_known_adducts
from ..filters.clean_adduct import CleanAdduct
from matchms.filtering.filters.derive_from_name_template import DeriveFromNameTemplate


logger = logging.getLogger("matchms")


class DeriveAdductFromName(DeriveFromNameTemplate):
    def __init__(self, remove_from_name: bool = True):
        super().__init__(remove_from_name, "adduct")

    def derive(self, name, spectrum):
        adduct_from_name = None
        name_split = name.split(" ")
        for name_part in name_split[::-1][:2]:
            if _looks_like_adduct(name_part):
                adduct_from_name = name_part
                break

        if adduct_from_name and self.remove_from_name:
            name_adduct_removed = " ".join([x for x in name_split if x != adduct_from_name])
            name_adduct_removed = name_adduct_removed.strip("; ")
            spectrum.set("compound_name", name_adduct_removed)
            logger.info("Removed adduct %s from compound name.", adduct_from_name)

        if adduct_from_name and not _looks_like_adduct(spectrum.get(self.metadata_key)):
            adduct_cleaned = CleanAdduct._clean_adduct(adduct_from_name)
            spectrum.set(self.metadata_key, adduct_cleaned)
            logger.info("Added adduct %s from the compound name to metadata.", spectrum.get(self.metadata_key))

        return spectrum


def _looks_like_adduct(adduct):
    """Return True if input string has expected format of an adduct."""
    if not isinstance(adduct, str):
        return False
    # Clean adduct
    adduct = CleanAdduct._clean_adduct(adduct)
    # Load lists of default known adducts
    known_adducts = load_known_adducts()
    if adduct in list(known_adducts["adduct"]):
        return True

    # Expect format like: "[2M-H]" or "[2M+Na]+"
    regexp1 = r"^\[(([0-4]M)|(M[0-9])|(M))((Br)|(Br81)|(Cl)|(Cl37)|(S)){0,}[+-][A-Z0-9\+\-\(\)aglire]{1,}[\]0-4+-]{1,4}"
    return re.search(regexp1, adduct) is not None