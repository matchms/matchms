from typing import Optional
import numpy as np
from matchms.Fragments import Fragments
from matchms.typing import SpectrumType


def select_by_intensity(
    spectrum_in: SpectrumType, intensity_from: float = 10.0, intensity_to: float = 200.0, clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """Keep only peaks within set intensity range (keep if
    intensity_from >= intensity >= intensity_to). In most cases it is adviced to
    use :py:func:`select_by_relative_intensity` function instead.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    intensity_from:
        Set lower threshold for peak intensity. Default is 10.0.
    intensity_to:
        Set upper threshold for peak intensity. Default is 200.0.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with peaks within the specified intensity range, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    assert intensity_from <= intensity_to, "'intensity_from' should be smaller than or equal to 'intensity_to'."

    condition = np.logical_and(intensity_from <= spectrum.peaks.intensities, spectrum.peaks.intensities <= intensity_to)

    spectrum.peaks = Fragments(mz=spectrum.peaks.mz[condition], intensities=spectrum.peaks.intensities[condition])

    return spectrum
