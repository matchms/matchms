import numpy
from ..Spikes import Spikes
from ..typing import SpectrumType


def select_by_intensity(spectrum_in: SpectrumType, intensity_from: float = 10.0,
                        intensity_to: float = 200.0) -> SpectrumType:
    """Keep only peaks within set intensity range (keep if
    intensity_from >= intensity >= intensity_to). In most cases it is adviced to
    use :py:func:`select_by_relative_intensity` function instead.

    Parameters
    ----------
    intensity_from:
        Set lower threshold for peak intensity. Default is 10.0.
    intensity_to:
        Set upper threshold for peak intensity. Default is 200.0.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    assert intensity_from <= intensity_to, "'intensity_from' should be smaller than or equal to 'intensity_to'."

    condition = numpy.logical_and(intensity_from <= spectrum.peaks.intensities,
                                  spectrum.peaks.intensities <= intensity_to)

    spectrum.peaks = Spikes(mz=spectrum.peaks.mz[condition],
                            intensities=spectrum.peaks.intensities[condition])

    return spectrum
