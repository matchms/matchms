from typing import Optional
import numpy as np
from matchms.filtering._dispatch import collection_filter
from matchms.Fragments import Fragments
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


def _validate_mz_range(mz_from: float, mz_to: float) -> None:
    if mz_from > mz_to:
        raise ValueError("'mz_from' should be smaller than or equal to 'mz_to'.")


def _select_by_mz_spectrum(
        spectrum_in: SpectrumType,
        mz_from: float = 0.0,
        mz_to: float = 1000.0,
        clone: Optional[bool] = True,
    ) -> Optional[SpectrumType]:
    """Keep only peaks between mz_from and mz_to.

    Peaks are kept if ``mz_from <= m/z <= mz_to``.

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
    """
    if spectrum_in is None:
        return None

    _validate_mz_range(mz_from, mz_to)

    spectrum = spectrum_in.clone() if clone else spectrum_in

    condition = np.logical_and(
        mz_from <= spectrum.peaks.mz,
        spectrum.peaks.mz <= mz_to,
    )

    spectrum.peaks = Fragments(
        mz=spectrum.peaks.mz[condition],
        intensities=spectrum.peaks.intensities[condition],
    )

    return spectrum


def _select_by_mz_collection(
        spectrum_in: SpectraCollection,
        mz_from: float = 0.0,
        mz_to: float = 1000.0,
        clone: Optional[bool] = True,
    ) -> SpectraCollection:
    """Keep only peaks between mz_from and mz_to for a SpectraCollection."""
    _validate_mz_range(mz_from, mz_to)

    target = spectrum_in.copy() if clone else spectrum_in

    target._fragments = target._fragments.slice_mz(
        mz_min=mz_from,
        mz_max=mz_to,
    )
    target._clear_cache(["fragment_hashes", "spectra_hashes"])

    return target


select_by_mz = collection_filter(
    _select_by_mz_spectrum,
    collection_impl=_select_by_mz_collection,
)
