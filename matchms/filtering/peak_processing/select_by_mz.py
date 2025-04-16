from typing import Optional
import numpy as np
from matchms.Fragments import Fragments
from matchms.typing import SpectrumType


def select_by_mz(
    spectrum_in: SpectrumType, mz_from: float = 0.0, mz_to: float = 1000.0, clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """Keep only peaks between mz_from and mz_to (keep if mz_from >= m/z >= mz_to).

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    mz_from:
        Set lower threshold for m/z peak positions. Default is 0.0.
    mz_to:
        Set upper threshold for m/z peak positions. Default is 1000.0.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with peaks within the specified mz range, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    assert mz_from <= mz_to, "'mz_from' should be smaller than or equal to 'mz_to'."

    condition = np.logical_and(mz_from <= spectrum.peaks.mz, spectrum.peaks.mz <= mz_to)

    spectrum.peaks = Fragments(mz=spectrum.peaks.mz[condition], intensities=spectrum.peaks.intensities[condition])

    return spectrum
