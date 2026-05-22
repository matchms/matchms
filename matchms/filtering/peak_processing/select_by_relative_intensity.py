from typing import Optional
import numpy as np
from matchms.filtering._dispatch import collection_filter
from matchms.Fragments import Fragments
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


def _validate_relative_intensity_range(
        intensity_from: float,
        intensity_to: float,
    ) -> None:
    if intensity_from < 0.0:
        raise ValueError("'intensity_from' should be larger than or equal to 0.")
    if intensity_to > 1.0:
        raise ValueError("'intensity_to' should be smaller than or equal to 1.0.")
    if intensity_from > intensity_to:
        raise ValueError(
            "'intensity_from' should be smaller than or equal to 'intensity_to'."
        )


def _select_by_relative_intensity_spectrum(
        spectrum_in: SpectrumType,
        intensity_from: float = 0.0,
        intensity_to: float = 1.0,
        clone: Optional[bool] = True
    ) -> Optional[SpectrumType]:
    """Keep only peaks within set relative intensity range (keep if
    intensity_from >= intensity >= intensity_to).

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    intensity_from:
        Set lower threshold for relative peak intensity. Default is 0.0.
    intensity_to:
        Set upper threshold for relative peak intensity. Default is 1.0.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with peaks within the relative intensity range, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    _validate_relative_intensity_range(intensity_from, intensity_to)

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if len(spectrum.peaks) > 0:
        scale_factor = np.max(spectrum.peaks.intensities)

        if scale_factor > 0:
            normalized_intensities = spectrum.peaks.intensities / scale_factor
            condition = np.logical_and(
                intensity_from <= normalized_intensities,
                normalized_intensities <= intensity_to,
            )
        else:
            condition = np.zeros(spectrum.peaks.intensities.shape, dtype=bool)

        spectrum.peaks = Fragments(
            mz=spectrum.peaks.mz[condition],
            intensities=spectrum.peaks.intensities[condition],
        )

    return spectrum


def _select_by_relative_intensity_collection(
    spectrum_in: SpectraCollection,
    intensity_from: float = 0.0,
    intensity_to: float = 1.0,
    clone: Optional[bool] = True,
) -> SpectraCollection:
    """Keep only peaks within set relative intensity range for a SpectraCollection."""
    _validate_relative_intensity_range(intensity_from, intensity_to)

    target = spectrum_in.copy() if clone else spectrum_in

    target._fragments = target._fragments.select_by_relative_intensity(
        intensity_from=intensity_from,
        intensity_to=intensity_to,
    )
    target._clear_cache(["fragment_hashes", "spectra_hashes"])

    return target


select_by_relative_intensity = collection_filter(
    _select_by_relative_intensity_spectrum,
    collection_impl=_select_by_relative_intensity_collection,
)
