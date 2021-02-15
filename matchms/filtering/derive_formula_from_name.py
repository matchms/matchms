import re
from ..typing import SpectrumType


def derive_formula_from_name(spectrum_in: SpectrumType,
                             remove_formula_from_name: bool = True) -> SpectrumType:
    """Detect and remove misplaced formula in compound name and add to metadata.

    Method to find misplaced formulas in compound name based on regular expression.
    This will not chemically test the detected formula, so the search is limited
    to frequently occuring types of shape 'C47H83N1O8P1'.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    remove_formula_from_name:
        Remove found formula from compound name if set to True. Default is True.
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

    # Detect formula at end of compound name
    end_of_name = name.split(" ")[-1]
    formula_from_name = end_of_name if looks_like_formula(end_of_name) else None

    if formula_from_name and remove_formula_from_name:
        name_formula_removed = " ".join(name.split(" ")[:-1])
        spectrum.set("compound_name", name_formula_removed)
        print("Removed formula {} from compound name.".format(formula_from_name))

    # Add found formula to metadata (if not present yet)
    if formula_from_name and spectrum.get("formula", None) is None:
        spectrum.set("formula", formula_from_name)
        print("Added formula {} to metadata.".format(formula_from_name))

    return spectrum


def looks_like_formula(formula):
    """Return True if input string has expected format of a molecular formula.
    Does only consider most frequent atoms found in many name strings.
    """
    regex_atoms = r"(C|F|H|N|O|P|S)"
    atom_count = len(re.findall(regex_atoms, formula))
    regexp = r"^(C|F|H|N|O|P|S|[0-9]|\(|\)){3,}$"
    return (atom_count > 2) and (re.search(regexp, formula) is not None)
