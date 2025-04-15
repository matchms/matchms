import logging
from typing import Optional
from matchms.Spectrum import Spectrum
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def require_maximum_number_of_peaks(
    spectrum_in: Spectrum, maximum_number_of_fragments: int = 1000, clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """Spectrum will be set to None when it has more peaks than maximum_number_of_fragments.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    maximum_number_of_fragments:
        Number of minimum required peaks. Spectra with fewer peaks will be set
        to 'None'.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Untouched Spectrum or 'None'.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if spectrum.peaks.intensities.size > maximum_number_of_fragments:
        logger.info(
            "Spectrum with %s (>%s) peaks was set to None.",
            str(spectrum.peaks.intensities.size),
            str(maximum_number_of_fragments),
        )
        return None

    return spectrum
