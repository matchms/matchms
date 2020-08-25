import numpy
from ..Spikes import Spikes
from ..typing import SpectrumType


def remove_peaks_around_precursor_mz(spectrum_in: SpectrumType, mz_tolerance: float = 17) -> SpectrumType:

    """Remove peaks that are within mz_tolerance (in Da) of
       the precursor mz, exlcuding the precursor peak.

    Parameters
    ----------
    spectrum
        Input spectrum.
    mz_tolerance
        Tolerance of mz values that are not allowed to lie
        within the precursor mz. Default is 17 Da.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    assert mz_tolerance >= 0, "mz_tolerance must be a positive floating point."
    precursor_mz = spectrum.get("precursor_mz")
    mzs, intensities = spectrum.peaks
    if precursor_mz:
        peaks_to_remove = ((numpy.abs(precursor_mz-mzs) <= mz_tolerance) & (mzs != precursor_mz))
        new_mzs, new_intensities = mzs[~peaks_to_remove], intensities[~peaks_to_remove]
        spectrum.peaks = Spikes(mz=new_mzs, intensities=new_intensities)

    return spectrum
