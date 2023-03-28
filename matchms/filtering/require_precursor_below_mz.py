import logging
from ..typing import SpectrumType


logger = logging.getLogger("matchms")


def require_precursor_below_mz(spectrum_in: SpectrumType, max_mz: float = 1000) -> SpectrumType:

    """Returns None if the precursor_mz of a spectrum is above
       max_mz.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    max_mz:
        Maximum mz value for the precursor mz of a spectrum.
        All precursor mz values greater or equal to this
        will return none. Default is 1000.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    precursor_mz = spectrum.get("precursor_mz", None)
    assert precursor_mz is not None, "Precursor mz absent."
    assert isinstance(precursor_mz, (float, int)), ("Expected 'precursor_mz' to be a scalar number.",
                                                    "Consider applying 'add_precursor_mz' filter first.")
    assert max_mz >= 0, "max_mz must be a positive scalar."
    if precursor_mz >= max_mz:
        logger.info("Spectrum with precursor_mz %s (>%s) was set to None.",
                    str(precursor_mz), str(max_mz))
        return None

    return spectrum
