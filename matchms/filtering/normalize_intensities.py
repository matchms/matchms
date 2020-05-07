import numpy
from matchms.typing import SpectrumType
from ..Spikes import Spikes


def normalize_intensities(spectrum_in: SpectrumType) -> SpectrumType:
    """Normalize intensities to unit height."""

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if len(spectrum.peaks) > 0:
        scale_factor = numpy.max(spectrum.peaks.intensities)
        mz, intensities = spectrum.peaks
        normalized_intensities = intensities / scale_factor
        spectrum.peaks = Spikes(mz=mz, intensities=normalized_intensities)

    return spectrum
