import numpy
from ..Spikes import Spikes
from ..typing import SpectrumType


def select_by_mz(spectrum_in: SpectrumType, mz_from: float = 0.0,
                 mz_to: float = 1000.0) -> SpectrumType:
    """Keep only peaks between mz_from and mz_to (keep if mz_from >= m/z >= mz_to).

    Parameters
    ----------
    mz_from:
        Set lower threshold for m/z peak positions. Default is 0.0.
    mz_to:
        Set upper threshold for m/z peak positions. Default is 1000.0.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    assert mz_from <= mz_to, "'mz_from' should be smaller than or equal to 'mz_to'."

    condition = numpy.logical_and(mz_from <= spectrum.peaks.mz, spectrum.peaks.mz <= mz_to)

    spectrum.peaks = Spikes(mz=spectrum.peaks.mz[condition],
                            intensities=spectrum.peaks.intensities[condition])

    return spectrum
