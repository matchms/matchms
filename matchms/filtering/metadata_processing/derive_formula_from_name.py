from matchms.typing import SpectrumType
from matchms.filtering.filters.derive_formula_from_name import DeriveFormulaFromName


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

    spectrum = DeriveFormulaFromName(remove_formula_from_name).process(spectrum_in)
    return spectrum
