from matchms.typing import SpectrumType
from matchms.filtering.filters.require_minimum_of_high_peaks import RequireMinimumOfHighPeaks


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

    spectrum = RequireMinimumOfHighPeaks(no_peaks, intensity_percent).process(spectrum_in)
    return spectrum
