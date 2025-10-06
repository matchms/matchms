import logging
from typing import Optional
import numpy as np
from matchms.typing import SpectrumType
from ...Fragments import Fragments


logger = logging.getLogger("matchms")


def normalize_intensities(
    spectrum_in: SpectrumType, clone: Optional[bool] = True, scaling: Optional[tuple[float, float]] = None
) -> Optional[SpectrumType]:
    """Normalize intensities of peaks to unit height.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.
    scaling:
        Optional tuple (min, max) to scale intensities to specific range.
        If None, normalizes to 0-1 range.

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

    # Scale intensities to specific range
    if scaling:
        if (
            not isinstance(scaling, tuple)
            or len(scaling) != 2
            or not all(isinstance(val, (int, float)) for val in scaling)
        ):
            raise ValueError("Expected 'scaling' to be a tuple of two numbers (int or float).")

        min_val, max_val = scaling
        if min_val > max_val:
            raise ValueError("Expected 'scaling' to be a tuple where the first value is smaller than the second.")

        scaled_intensities = np.interp(
            normalized_intensities, (normalized_intensities.min(), normalized_intensities.max()), (min_val, max_val)
        )
    else:
        scaled_intensities = normalized_intensities

    spectrum.peaks = Fragments(mz=mz, intensities=scaled_intensities)
    return spectrum
