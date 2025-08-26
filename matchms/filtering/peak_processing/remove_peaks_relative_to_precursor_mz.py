from typing import Optional
import numpy as np
from matchms.Fragments import Fragments
from matchms.typing import SpectrumType


def remove_peaks_relative_to_precursor_mz(
    spectrum_in: SpectrumType, relative_to_precursor: float = -1.6,
    clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """Remove peaks that are within mz_tolerance (in Da) of
       the precursor mz, exlcuding the precursor peak.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    relative_to_precursor:
        All peaks with mz values > precursor_mz + relative_to_precursor will be removed.
        Default is -1.6 Da based Flash Entropy article by Li and Fiehn, 2023, Nat. Comm.
        (see https://www.nature.com/articles/s41592-023-02012-9)
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
    if precursor_mz is None:
        raise ValueError("Undefined 'precursor_mz'.")
    if not isinstance(precursor_mz, (float, int)):
        raise ValueError(
            "Expected 'precursor_mz' to be a scalar number.",
            "Consider applying 'add_precursor_mz' filter first."
            )

    mzs, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
    peaks_to_remove = mzs > (precursor_mz + relative_to_precursor)
    new_mzs, new_intensities = mzs[~peaks_to_remove], intensities[~peaks_to_remove]
    spectrum.peaks = Fragments(mz=new_mzs, intensities=new_intensities)

    return spectrum
