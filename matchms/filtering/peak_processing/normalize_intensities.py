import logging
from typing import Optional
import numpy as np
from matchms.typing import SpectrumType
from ...Fragments import Fragments


logger = logging.getLogger("matchms")


def normalize_intensities(spectrum_in: SpectrumType, clone: Optional[bool] = True) -> Optional[SpectrumType]:
    """Normalize intensities of peaks to unit height.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with mormalized Intensities, or `None` if not present.
    """

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if len(spectrum.peaks) == 0:
        return spectrum

    max_intensity = np.max(spectrum.peaks.intensities)

    if max_intensity <= 0:
        logger.warning("Peaks of spectrum with all peak intensities <= 0 were deleted.")
        spectrum.peaks = Fragments(mz=np.array([]), intensities=np.array([]))
        return spectrum

    # Normalize peak intensities
    mz, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
    normalized_intensities = intensities / max_intensity
    spectrum.peaks = Fragments(mz=mz, intensities=normalized_intensities)

    return spectrum
