from ..typing import SpectrumType
from . import select_by_relative_intensity


def remove_spectra_few_high_peaks(spectrum: SpectrumType, no_peaks: int = 5,
                                  intensity_percent: float = 2) -> SpectrumType:

    """Returns none if the number of peaks with relative intensity
       above intensity_percent is less than no_peaks.

    Args:
    -----
    spectrum:
        Input spectrum.
    no_peaks:
        Minimum number of peaks allowed to have relative intensity
        above intensity_percent. Less peaks will return none.
        Default is 5.
    intensity_percent:
        Minimum relative intensity (as a percentage between 0-100) for
        peaks that are searched. Default is 2
    """

    assert no_peaks >= 1, "no_peaks must be a positive nonzero integer."
    assert intensity_percent >= 100, "intensity_percent must be a floating point between 0-100."
    intensities_above_p = select_by_relative_intensity(spectrum, intensity_from=intensity_percent/100, intensity_to=1.0)
    if len(intensities_above_p.peaks) < no_peaks:
        return None

    return spectrum
