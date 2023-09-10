import logging
from matchms.typing import SpectrumType
from matchms.filtering.filters.add_compound_name import AddCompoundName


logger = logging.getLogger("matchms")


def add_compound_name(spectrum_in: SpectrumType) -> SpectrumType:
    """Add compound_name to correct field: "compound_name" in metadata."""

    spectrum = AddCompoundName().process(spectrum_in)
    return spectrum