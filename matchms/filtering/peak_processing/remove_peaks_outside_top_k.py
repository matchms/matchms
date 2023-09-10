from matchms.typing import SpectrumType
from matchms.filtering.filters.remove_peaks_outside_top_k import RemovePeaksOutsideTopK


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

    spectrum = RemovePeaksOutsideTopK(k, mz_window).process(spectrum_in)
    return spectrum
