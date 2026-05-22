import logging
from typing import Optional
from matchms.filtering._dispatch import collection_filter
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType
from .select_by_relative_intensity import select_by_relative_intensity


logger = logging.getLogger("matchms")


def _validate_minimum_high_peaks_parameters(
    no_peaks: int,
    intensity_percent: float,
) -> None:
    if no_peaks < 1:
        raise ValueError("no_peaks must be a positive nonzero integer.")
    if not 0 <= intensity_percent <= 100:
        raise ValueError("intensity_percent must be a scalar between 0-100.")


def _require_minimum_number_of_high_peaks_spectrum(
    spectrum_in: SpectrumType,
    no_peaks: int = 5,
    intensity_percent: float = 2.0,
    clone: Optional[bool] = True,
) -> Optional[SpectrumType]:
    """Removes spectra if the number of peaks with relative intensity
    above or equal to intensity_percent is less than no_peaks.

    For single Spectrum import this will return 'None' when the number of peaks with relative intensity
    above or equal to intensity_percent is less than no_peaks.
    For SpectraCollection import, spectra with fewer peaks with relative intensity above or equal to
    intensity_percent than no_peaks will be removed from the collection.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    no_peaks:
        Minimum number of peaks allowed to have relative intensity
        above intensity_percent. Less peaks will return none.
        Default is 5.
    intensity_percent:
        Minimum relative intensity (as a percentage between 0-100) for
        peaks that are searched. Default is 2.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Untouched Spectrum or 'None'.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    _validate_minimum_high_peaks_parameters(no_peaks, intensity_percent)

    intensities_above_p = select_by_relative_intensity(
        spectrum,
        intensity_from=intensity_percent / 100,
        intensity_to=1.0
    )
    if len(intensities_above_p.peaks) < no_peaks:
        logger.info(
            "Spectrum with %s (<%s) peaks was set to None.",
            str(len(intensities_above_p.peaks)),
            str(no_peaks)
        )
        return None

    return spectrum


def _require_minimum_number_of_high_peaks_collection(
    spectrum_in: SpectraCollection,
    no_peaks: int = 5,
    intensity_percent: float = 2.0,
    clone: Optional[bool] = True,
) -> SpectraCollection | None:
    """Drop spectra with fewer than no_peaks high relative-intensity peaks."""
    _validate_minimum_high_peaks_parameters(no_peaks, intensity_percent)

    high_peak_counts = spectrum_in.fragments.count_peaks_above_relative_intensity(
        intensity_from=intensity_percent / 100,
    )

    keep_mask = high_peak_counts >= no_peaks

    if not keep_mask.any():
        return None

    return spectrum_in.filter(keep_mask, inplace=not clone)


require_minimum_number_of_high_peaks = collection_filter(
    _require_minimum_number_of_high_peaks_spectrum,
    collection_impl=_require_minimum_number_of_high_peaks_collection,
)
