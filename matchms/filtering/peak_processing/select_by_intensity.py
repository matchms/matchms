from typing import Optional
import numpy as np
from matchms.filtering._dispatch import collection_filter
from matchms.Fragments import Fragments
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


def _select_by_intensity_spectrum(
        spectrum_in: SpectrumType,
        intensity_from: float = 0.01,
        intensity_to: float = 1.0,
        clone: Optional[bool] = True
    ) -> Optional[SpectrumType]:
    """Keep only peaks within set intensity range (keep if
    intensity_from >= intensity >= intensity_to). In most cases it is adviced to
    use :py:func:`select_by_relative_intensity` function instead.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    intensity_from:
        Set lower threshold for peak intensity. Default is 0.01.
    intensity_to:
        Set upper threshold for peak intensity. Default is 1.0.
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

    if intensity_from > intensity_to:
        raise ValueError(
            "'intensity_from' should be smaller than or equal to 'intensity_to'."
        )

    condition = np.logical_and(
        intensity_from <= spectrum.peaks.intensities,
        spectrum.peaks.intensities <= intensity_to,
    )

    spectrum.peaks = Fragments(
        mz=spectrum.peaks.mz[condition],
        intensities=spectrum.peaks.intensities[condition],
    )

    return spectrum


def _select_by_intensity_collection(
    spectrum_in: SpectraCollection,
    intensity_from: float = 0.01,
    intensity_to: float = 1.0,
    clone: Optional[bool] = True,
) -> SpectraCollection:
    """Keep only peaks within set intensity range for a SpectraCollection."""
    if intensity_from > intensity_to:
        raise ValueError(
            "'intensity_from' should be smaller than or equal to 'intensity_to'."
        )

    target = spectrum_in.copy() if clone else spectrum_in

    target._fragments = target._fragments.select_by_intensity(
        intensity_from=intensity_from,
        intensity_to=intensity_to,
    )
    target._clear_cache(["fragment_hashes", "spectra_hashes"])

    return target


select_by_intensity = collection_filter(
    _select_by_intensity_spectrum,
    collection_impl=_select_by_intensity_collection,
)
