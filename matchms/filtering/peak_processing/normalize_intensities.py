from matchms.typing import SpectrumType
from matchms.filtering.filters.normalize_intensities import NormalizeIntensities


def normalize_intensities(spectrum_in: SpectrumType) -> SpectrumType:
    """Normalize intensities of peaks (and losses) to unit height."""

    spectrum = NormalizeIntensities().process(spectrum_in)
    return spectrum
