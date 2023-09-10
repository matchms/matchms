import logging
import re
from matchms.typing import SpectrumType
from matchms.filtering.filters.derive_from_name_template import DeriveFromNameTemplate


logger = logging.getLogger("matchms")


class DeriveFormulaFromName(DeriveFromNameTemplate):
    def __init__(self, remove_from_name: bool = True):
        super().__init__(remove_from_name, "formula")

    def derive(self, name, spectrum):
        end_of_name = name.split(" ")[-1]
        formula_from_name = end_of_name if _looks_like_formula(end_of_name) else None

        if formula_from_name and self.remove_from_name:
            name_formula_removed = " ".join(name.split(" ")[:-1])
            spectrum.set("compound_name", name_formula_removed)
            logger.info("Added formula %s to metadata.", formula_from_name)

        if formula_from_name and spectrum.get(self.metadata_key) is None:
            spectrum.set(self.metadata_key, formula_from_name)
            logger.info("Added formula %s to metadata.", formula_from_name)

        return spectrum


def _looks_like_formula(formula):
    """Return True if input string has expected format of a molecular formula.
    Does only consider most frequent atoms found in many name strings.
    """
    regex_atoms = r"([CFHNOPS])"
    atom_count = len(re.findall(regex_atoms, formula))
    regexp = r"^([CFHNOPS]|[0-9]|\(|\)){3,}$"
    return (atom_count > 2) and (re.search(regexp, formula) is not None)