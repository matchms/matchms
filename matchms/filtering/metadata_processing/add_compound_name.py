import logging
from typing import Optional

from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def add_compound_name(spectrum_in: SpectrumType, clone: Optional[bool] = True) -> SpectrumType:
    """Add compound_name to correct field: "compound_name" in metadata."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if spectrum.get("compound_name", None) is None:
        if isinstance(spectrum.get("name", None), str):
            spectrum.set("compound_name", spectrum.get("name"))
            return spectrum

        if isinstance(spectrum.get("title", None), str):
            spectrum.set("compound_name", spectrum.get("title"))
            return spectrum

        logger.info("No compound name found in metadata.")

    return spectrum
