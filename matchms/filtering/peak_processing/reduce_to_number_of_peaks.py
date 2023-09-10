from typing import Optional
import numpy as np
from matchms.typing import SpectrumType
from matchms.filtering.filters.reduce_to_number_of_peaks import ReduceToNumberOfPeaks


def reduce_to_number_of_peaks(spectrum_in: SpectrumType, n_required: int = 1, n_max: int = np.inf,
                              ratio_desired: Optional[float] = None) -> SpectrumType:
    """Lowest intensity peaks will be removed when it has more peaks than desired.

    Parameters
    ----------
    spectrum_in
        Input spectrum.
    n_required:
        Number of minimum required peaks. Spectra with fewer peaks will be set
        to 'None'. Default is 1.
    n_max:
        Maximum number of peaks. Remove peaks if more peaks are found. Default is inf.
    ratio_desired:
        Set desired ratio between maximum number of peaks and parent mass.
        For spectra without parent mass (e.g. GCMS spectra) this will raise an
        error when ratio_desired is used.
        Default is None.
    """

    spectrum = ReduceToNumberOfPeaks(n_required, n_max, ratio_desired).process(spectrum_in)
    return spectrum