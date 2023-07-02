import logging
from ..typing import SpectrumType
from .select_by_relative_intensity import select_by_relative_intensity


logger = logging.getLogger("matchms")


def require_minimum_of_high_peaks(spectrum_in: SpectrumType, no_peaks: int = 5,
                                  intensity_percent: float = 2.0) -> SpectrumType:

    """Returns None if the number of peaks with relative intensity
       above or equal to intensity_percent is less than no_peaks.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    no_peaks:
        Minimum number of peaks allowed to have relative intensity
        above intensity_percent. Less peaks will return none.
        Default is 5.
    intensity_percent:
        Minimum relative intensity (as a percentage between 0-100) for
        peaks that are searched. Default is 2
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    assert no_peaks >= 1, "no_peaks must be a positive nonzero integer."
    assert 0 <= intensity_percent <= 100, "intensity_percent must be a scalar between 0-100."
    intensities_above_p = select_by_relative_intensity(spectrum, intensity_from=intensity_percent/100, intensity_to=1.0)
    if len(intensities_above_p.peaks) < no_peaks:
        logger.info("Spectrum with %s (<%s) peaks was set to None.",
                    str(len(intensities_above_p.peaks)), str(no_peaks))
        return None

    return spectrum
