import numpy as np
from ..Fragments import Fragments
from ..typing import SpectrumType


def remove_peaks_outside_top_k(spectrum_in: SpectrumType, k: int = 6,
                               mz_window: float = 50) -> SpectrumType:

    """Remove all peaks which are not within *mz_window* of at least one
       of the *k* highest intensity peaks of the spectrum.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    k:
        The number of most intense peaks to compare to. Default is 6.
    mz_window:
        Window of mz values (in Da) that are allowed to lie within
        the top k peaks. Default is 50 Da.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    assert k >= 1, "k must be a positive nonzero integer."
    assert mz_window >= 0, "mz_window must be a positive scalar."
    mzs, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
    top_k = intensities.argsort()[::-1][0:k]
    k_ordered_mzs = mzs[top_k]
    indices = [i for i in range(len(mzs)) if i not in top_k]
    keep_idx = top_k.tolist()
    for i in indices:

        compare = abs(mzs[i]-k_ordered_mzs) <= mz_window
        if np.any(compare):
            keep_idx.append(i)

    keep_idx.sort()
    new_mzs, new_intensities = mzs[keep_idx], intensities[keep_idx]
    spectrum.peaks = Fragments(mz=new_mzs, intensities=new_intensities)

    return spectrum
