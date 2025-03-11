import logging
import numpy as np
from matchms.typing import SpectrumType
from ...Fragments import Fragments


logger = logging.getLogger("matchms")


def normalize_intensities(spectrum_in: SpectrumType) -> SpectrumType:
    """Normalize intensities of peaks to unit height."""

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if len(spectrum.peaks) == 0:
        return spectrum

    max_intensity = np.max(spectrum.peaks.intensities)

    if max_intensity <= 0:
        logger.warning("Peak intensities of spectrum with all peak intensities <= 0 were set to 0.")
        zero_intensities = np.zeros(spectrum.peaks.intensities.shape)
        spectrum.peaks = Fragments(mz=spectrum.peaks.mz, intensities=zero_intensities)
        return spectrum

    # Normalize peak intensities
    mz, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
    normalized_intensities = intensities / max_intensity
    spectrum.peaks = Fragments(mz=mz, intensities=normalized_intensities)

    return spectrum
