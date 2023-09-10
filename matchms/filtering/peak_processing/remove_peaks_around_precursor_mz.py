from matchms.typing import SpectrumType
from matchms.filtering.filters.remove_peaks_around_precursor_mz import RemovePeaksAroundPrecursorMz


def remove_peaks_around_precursor_mz(spectrum_in: SpectrumType, mz_tolerance: float = 17) -> SpectrumType:

    """Remove peaks that are within mz_tolerance (in Da) of
       the precursor mz, exlcuding the precursor peak.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    mz_tolerance:
        Tolerance of mz values that are not allowed to lie
        within the precursor mz. Default is 17 Da.
    """

    spectrum = RemovePeaksAroundPrecursorMz(mz_tolerance).process(spectrum_in)
    return spectrum
