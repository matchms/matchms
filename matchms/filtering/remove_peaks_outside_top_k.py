import numpy
from ..Spikes import Spikes
from ..typing import SpectrumType


def remove_peaks_outside_top_k(spectrum: SpectrumType, k: int = 6,
                               mz_window: float = 50) -> SpectrumType:

    """Keep peaks only if they are in the top k intense peaks
       within a desired mz (Da) window

    Args:
    -----
    spectrum:
        Input spectrum.
    k:
        The number of most intense peaks to compare to. Default is 6.
    mz_window:
        Window of mz values (in Da) that are allowed to lie within
        the top k peaks. Default is 50 Da.
    """

    assert k >= 1, "k must be a positive nonzero integer."
    assert mz_window >= 0, "mz_window must be a positive floating point."
    mzs, intensities = spectrum.peaks
    top_k = intensities.argsort()[::-1][0:k]
    k_ordered_mzs = mzs[top_k]
    indices = [i for i in range(len(mzs)) if i not in top_k]
    new_mzs, new_intensities = mzs, intensities
    for i in indices:

        compare = abs(mzs[i]-k_ordered_mzs) <= mz_window
        if not numpy.any(compare):
            new_mzs[i], new_intensities[i] = numpy.nan, numpy.nan

    nans = numpy.isnan(new_mzs)
    new_mzs, new_intensities = new_mzs[~nans], new_intensities[~nans]
    spectrum.peaks = Spikes(mz=new_mzs, intensities=new_intensities)

    return spectrum
