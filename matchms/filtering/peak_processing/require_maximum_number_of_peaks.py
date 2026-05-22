import logging
from typing import Optional
from matchms.filtering._dispatch import collection_filter
from matchms.SpectraCollection import SpectraCollection
from matchms.Spectrum import Spectrum
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _require_maximum_number_of_peaks_spectrum(
        spectrum_in: Spectrum,
        maximum_number_of_fragments: int = 1000,
        clone: Optional[bool] = True,
    ) -> Optional[SpectrumType]:
    """Spectrum will be removed when it has more peaks than maximum_number_of_fragments.

    For single Spectrum import this will return 'None' when the number of peaks exceeds the
    maximum_number_of_fragments. For SpectraCollection import, spectra with more peaks than 
    maximum_number_of_fragments will be removed from the collection.
    
    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    maximum_number_of_fragments:
        Number of minimum required peaks. Spectra with fewer peaks will be set
        to 'None'.
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

    if spectrum.peaks.intensities.size > maximum_number_of_fragments:
        logger.info(
            "Spectrum with %s (>%s) peaks was set to None.",
            str(spectrum.peaks.intensities.size),
            str(maximum_number_of_fragments),
        )
        return None

    return spectrum


def _require_maximum_number_of_peaks_collection(
    spectrum_in: SpectraCollection,
    maximum_number_of_fragments: int = 1000,
    clone: Optional[bool] = True,
) -> SpectraCollection | None:
    """Drop spectra with more peaks than maximum_number_of_fragments."""
    peak_counts = spectrum_in.fragments.count(axis=1)
    keep_mask = peak_counts <= maximum_number_of_fragments

    if not keep_mask.any():
        return None

    return spectrum_in.filter(keep_mask, inplace=not clone)


require_maximum_number_of_peaks = collection_filter(
    _require_maximum_number_of_peaks_spectrum,
    collection_impl=_require_maximum_number_of_peaks_collection,
)
