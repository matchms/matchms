import numpy
from ..Spikes import Spikes
from ..typing import SpectrumType


def select_by_mz(spectrum_in: SpectrumType, mz_from=0.0, mz_to=1000.0) -> SpectrumType:

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    assert mz_from <= mz_to, "'mz_from' should be smaller than or equal to 'mz_to'."

    condition = numpy.logical_and(mz_from <= spectrum.peaks.mz, spectrum.peaks.mz <= mz_to)

    spectrum.peaks = Spikes(mz=spectrum.peaks.mz[condition],
                            intensities=spectrum.peaks.intensities[condition])

    return spectrum
