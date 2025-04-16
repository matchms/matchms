from typing import Optional
import numpy as np
from matchms.Fragments import Fragments
from matchms.typing import SpectrumType


def remove_peaks_around_precursor_mz(
    spectrum_in: SpectrumType, mz_tolerance: float = 17, clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """Remove peaks that are within mz_tolerance (in Da) of
       the precursor mz, exlcuding the precursor peak.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    mz_tolerance:
        Tolerance of mz values that are not allowed to lie
        within the precursor mz. Default is 17 Da.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with removed peaks, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    precursor_mz = spectrum.get("precursor_mz", None)
    assert precursor_mz is not None, "Precursor mz absent."
    assert isinstance(precursor_mz, (float, int)), (
        "Expected 'precursor_mz' to be a scalar number.",
        "Consider applying 'add_precursor_mz' filter first.",
    )
    assert mz_tolerance >= 0, "mz_tolerance must be a positive scalar."

    mzs, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
    peaks_to_remove = (np.abs(precursor_mz - mzs) <= mz_tolerance) & (mzs != precursor_mz)
    new_mzs, new_intensities = mzs[~peaks_to_remove], intensities[~peaks_to_remove]
    spectrum.peaks = Fragments(mz=new_mzs, intensities=new_intensities)

    return spectrum
