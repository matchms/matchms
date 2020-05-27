from math import ceil
from ..Spikes import Spikes
from ..typing import SpectrumType


def reduce_to_number_of_peaks(spectrum_in: SpectrumType, n_required=1, n_max=100,
                              ratio_desired=None) -> SpectrumType:
    """Lowest intensity peaks will be removed when it has more peaks than desired.

    Args:
    ----
    spectrum_in: matchms.Spectrum()
        Input spectrum.
    n_required: int
        Number of minimum required peaks. Spectra with fewer peaks will be set
        to 'None'. Default is 1.
    n_max: int
        Maximum number of peaks. Remove peaks if more peaks are found.
    ratio_desired: float, optional
        Set desired ratio between maximum number of peaks and parent mass.
        Default is None.
    """
    def _set_maximum_number_of_peaks_to_keep():
        parent_mass = spectrum.get("parent_mass", None)
        if parent_mass and ratio_desired:
            n_desired_by_mass = int(ceil(ratio_desired * parent_mass))
            return max(n_required, n_desired_by_mass)
        return n_max

    def _remove_lowest_intensity_peaks():
        mz, intensities = spectrum.peaks
        idx = intensities.argsort()[-threshold:]
        idx_sort_by_mz = mz[idx].argsort()
        spectrum.peaks = Spikes(mz=mz[idx][idx_sort_by_mz],
                                intensities=intensities[idx][idx_sort_by_mz])

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.peaks.intensities.size < n_required:
        return None

    threshold = _set_maximum_number_of_peaks_to_keep()
    if spectrum.peaks.intensities.size < threshold:
        return spectrum

    _remove_lowest_intensity_peaks()

    return spectrum
