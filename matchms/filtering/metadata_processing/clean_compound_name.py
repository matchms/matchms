from matchms.typing import SpectrumType
from matchms.filtering.filters.clean_compound_name import CleanCompoundName


def clean_compound_name(spectrum_in: SpectrumType) -> SpectrumType:
    """Clean compound name.

    A list of frequently seen name additions that do not belong to the compound
    name will be removed.
    """

    spectrum = CleanCompoundName().process(spectrum_in)
    return spectrum
