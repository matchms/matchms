import numpy as np
from matchms.filtering._dispatch import collection_filter
from matchms.Fragments import Fragments
from matchms.typing import SpectrumType


def _remove_peaks_around_precursor_mz(
        spectrum_in: SpectrumType, mz_tolerance: float = 17, clone: bool | None = True
    ) -> SpectrumType | None:
    """Remove peaks that are within mz_tolerance (in Da) of
       the precursor mz, excluding the precursor peak.

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

    if mz_tolerance < 0:
        raise ValueError("mz_tolerance must be a positive scalar.")

    spectrum = spectrum_in.clone() if clone else spectrum_in

    precursor_mz = spectrum.get("precursor_mz", None)
    if precursor_mz is None:
        raise ValueError("Undefined 'precursor_mz'.")

    if not isinstance(precursor_mz, (float, int, np.floating, np.integer)):
        raise ValueError(
            "Expected 'precursor_mz' to be a scalar number. "
            "Consider applying 'add_precursor_mz' filter first."
        )

    precursor_mz = float(precursor_mz)

    mzs, intensities = spectrum.peaks.mz, spectrum.peaks.intensities

    is_near_precursor = np.abs(precursor_mz - mzs) <= mz_tolerance

    # Important for SpectraCollection fallback:
    # reconstructed m/z values are bin centers and may not be exactly equal to
    # the original precursor_mz.
    is_precursor_peak = np.isclose(
        mzs,
        precursor_mz,
        rtol=0.0,
        atol=1e-6,
    )

    peaks_to_remove = is_near_precursor & ~is_precursor_peak

    spectrum.peaks = Fragments(
        mz=mzs[~peaks_to_remove],
        intensities=intensities[~peaks_to_remove],
    )

    return spectrum


# wrapper
remove_peaks_around_precursor_mz = collection_filter(
    _remove_peaks_around_precursor_mz,
    collection_impl=None,
)
