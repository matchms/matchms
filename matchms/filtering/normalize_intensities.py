import logging
import numpy as np
from matchms.typing import SpectrumType
from ..Fragments import Fragments


logger = logging.getLogger("matchms")


def normalize_intensities(spectrum_in: SpectrumType) -> SpectrumType:
    """Normalize intensities of peaks (and losses) to unit height."""

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if len(spectrum.peaks) == 0:
        return spectrum

    max_intensity = np.max(spectrum.peaks.intensities)

    if max_intensity <= 0:
        logger.warning("Spectrum with all peak intensities <= 0 was set to None.")
        return None

    # Normalize peak intensities
    mz, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
    normalized_intensities = intensities / max_intensity
    spectrum.peaks = Fragments(mz=mz, intensities=normalized_intensities)

    # Normalize loss intensities
    if spectrum.losses is not None and len(spectrum.losses) > 0:
        mz, intensities = spectrum.losses.mz, spectrum.losses.intensities
        normalized_intensities = intensities / max_intensity
        spectrum.losses = Fragments(mz=mz, intensities=normalized_intensities)

    return spectrum
