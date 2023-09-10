from matchms.typing import SpectrumType
from matchms.filtering.filters.require_precursor_below_mz import RequirePrecursorBelowMz


def require_precursor_below_mz(spectrum_in: SpectrumType, max_mz: float = 1000) -> SpectrumType:

    """Returns None if the precursor_mz of a spectrum is above
       max_mz.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    max_mz:
        Maximum mz value for the precursor mz of a spectrum.
        All precursor mz values greater or equal to this
        will return none. Default is 1000.
    """

    spectrum = RequirePrecursorBelowMz(max_mz).process(spectrum_in)
    return spectrum
