import logging
from math import ceil
from typing import Optional
import numpy as np
from ..Fragments import Fragments
from ..typing import SpectrumType


logger = logging.getLogger("matchms")


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
    def _set_maximum_number_of_peaks_to_keep():
        parent_mass = spectrum.get("parent_mass", None)
        if parent_mass and ratio_desired:
            n_desired_by_mass = int(ceil(ratio_desired * parent_mass))
            return min(max(n_required, n_desired_by_mass), n_max)
        if not ratio_desired:
            return n_max
        raise ValueError("Cannot use ratio_desired for spectrum without parent_mass.")

    def _remove_lowest_intensity_peaks():
        mz, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
        idx = intensities.argsort()[-threshold:]
        idx_sort_by_mz = mz[idx].argsort()
        spectrum.peaks = Fragments(mz=mz[idx][idx_sort_by_mz],
                                   intensities=intensities[idx][idx_sort_by_mz])

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.peaks.intensities.size < n_required:
        logger.info("Spectrum with %s (<%s) peaks was set to None.",
                    str(spectrum.peaks.intensities.size), str(n_required))
        return None

    threshold = _set_maximum_number_of_peaks_to_keep()
    if spectrum.peaks.intensities.size < threshold:
        return spectrum

    _remove_lowest_intensity_peaks()

    return spectrum
