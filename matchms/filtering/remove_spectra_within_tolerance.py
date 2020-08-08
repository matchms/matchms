from ..Spikes import Spikes
from ..typing import SpectrumType
import numpy


def remove_spectra_within_tolerance(spectrum: SpectrumType, mz_tolerance: float = 17) -> SpectrumType:

    """Remove peaks that are within mz_tolerance (in Da) of
       the precursor mz, exlcuding the precursor peak

    Args:
    -----
    spectrum:
        Input spectrum.
    mz_tolerance:
        Tolerance of mz values that are not allowed to lie
        within the precursor mz. Default is 17 Da.

    """

    assert mz_tolerance >= 0, "mz_tolerance must be a positive floating point."
    precursor_mz = spectrum.get("precursor_mz")
    mzs, intensities = spectrum.peaks
    new_mzs, new_intensities = mzs, intensities
    if precursor_mz:
        for i, mz in enumerate(mzs):

            if abs(precursor_mz-mz) <= mz_tolerance and mz != precursor_mz:
                new_mzs[i], new_intensities[i] = numpy.nan, numpy.nan

        nans = numpy.isnan(new_mzs)
        new_mzs, new_intensities = new_mzs[~nans], new_intensities[~nans]
        spectrum.peaks = Spikes(mz=new_mzs, intensities=new_intensities)

    return spectrum
