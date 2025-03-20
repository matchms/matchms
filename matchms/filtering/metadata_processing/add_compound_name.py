import logging
from typing import Optional
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def add_compound_name(spectrum_in: SpectrumType, clone: Optional[bool] = True) -> Optional[SpectrumType]:
    """Add compound_name to correct field: "compound_name" in metadata.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with added compound name, or `None` if not present.
    """
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
