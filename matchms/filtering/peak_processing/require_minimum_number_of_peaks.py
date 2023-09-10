from typing import Optional
from matchms.typing import SpectrumType
from matchms.filtering.filters.require_minimum_number_of_peaks import RequireMinimumNumberOfPeaks


def require_minimum_number_of_peaks(spectrum_in: SpectrumType,
                                    n_required: int = 10,
                                    ratio_required: Optional[float] = None) -> SpectrumType:
    """Spectrum will be set to None when it has fewer peaks than required.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    n_required:
        Number of minimum required peaks. Spectra with fewer peaks will be set
        to 'None'.
    ratio_required:
        Set desired ratio between minimum number of peaks and parent mass.
        Default is None.

    """

    spectrum = RequireMinimumNumberOfPeaks(n_required, ratio_required).process(spectrum_in)
    return spectrum
