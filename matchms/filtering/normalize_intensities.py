import numpy
from matchms.typing import SpectrumType
from ..Spikes import Spikes


def normalize_intensities(spectrum_in: SpectrumType) -> SpectrumType:
    """Normalize intensities of peaks (and losses) to unit height."""

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if len(spectrum.peaks) == 0:
        return spectrum

    max_intensity = numpy.max(spectrum.peaks.intensities)

    # Normalize peak intensities
    mz, intensities = spectrum.peaks
    normalized_intensities = intensities / max_intensity
    spectrum.peaks = Spikes(mz=mz, intensities=normalized_intensities)

    # Normalize loss intensities
    if spectrum.losses is not None and len(spectrum.losses) > 0:
        mz, intensities = spectrum.losses
        normalized_intensities = intensities / max_intensity
        spectrum.losses = Spikes(mz=mz, intensities=normalized_intensities)

    return spectrum
