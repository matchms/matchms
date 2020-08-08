from ..typing import SpectrumType


def remove_spectra_below(spectrum: SpectrumType, max_mz: float = 1000) -> SpectrumType:

    """Returns none if the precursor_mz of a spectrum is above
       max_mz.
    Args:
    -----
    spectrum:
        Input spectrum.
    max_mz:
        Maximum mz value for the precursor mz of a spectrum.
        All precursor mz values greater or equal to this
        will return none. Default is 1000.

    """

    assert max_mz >= 0, "max_mz must be a positive floating point."
    precursor_mz = spectrum.get("precursor_mz")
    if precursor_mz >= max_mz:
        return None

    return spectrum
