import numpy
from ..Spikes import Spikes
from ..typing import SpectrumType


def select_by_relative_intensity(spectrum_in: SpectrumType, intensity_from: float = 0.0,
                                 intensity_to: float = 1.0) -> SpectrumType:
    """Keep only peaks within set relative intensity range (keep if
    intensity_from >= intensity >= intensity_to).

    Parameters
    ----------
    intensity_from:
        Set lower threshold for relative peak intensity. Default is 0.0.
    intensity_to:
        Set upper threshold for relative peak intensity. Default is 1.0.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    assert intensity_from >= 0.0, "'intensity_from' should be larger than or equal to 0."
    assert intensity_to <= 1.0, "'intensity_to' should be smaller than or equal to 1.0."
    assert intensity_from <= intensity_to, "'intensity_from' should be smaller than or equal to 'intensity_to'."

    if len(spectrum.peaks) > 0:
        scale_factor = numpy.max(spectrum.peaks.intensities)
        normalized_intensities = spectrum.peaks.intensities / scale_factor
        condition = numpy.logical_and(intensity_from <= normalized_intensities, normalized_intensities <= intensity_to)
        spectrum.peaks = Spikes(mz=spectrum.peaks.mz[condition],
                                intensities=spectrum.peaks.intensities[condition])

    return spectrum
