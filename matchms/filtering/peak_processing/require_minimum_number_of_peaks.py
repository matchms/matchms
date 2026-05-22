import logging
from math import ceil
import numpy as np
import pandas as pd
from matchms.filtering._dispatch import collection_filter
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _require_minimum_number_of_peaks_spectrum(
    spectrum_in: SpectrumType,
    n_required: int = 10,
    ratio_required: float | None = None,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Spectrum will be set to None when it has fewer peaks than required.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    n_required:
        Number of minimum required peaks. Spectra with fewer peaks will be set
        to 'None'.
    ratio_required:
        Set desired ratio between minimum number of peaks and parent mass.
        Default is None.
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

    parent_mass = spectrum.get("parent_mass", None)
    if parent_mass and ratio_required:
        n_required_by_mass = int(ceil(ratio_required * parent_mass))
        threshold = max(n_required, n_required_by_mass)
    else:
        threshold = n_required

    if spectrum.peaks.intensities.size < threshold:
        logger.info(
            "Spectrum with %s (<%s) peaks was set to None.",
            str(spectrum.peaks.intensities.size),
            str(threshold),
        )
        return None

    return spectrum


def _require_minimum_number_of_peaks_collection(
    spectrum_in: SpectraCollection,
    n_required: int = 10,
    ratio_required: float | None = None,
    clone: bool | None = True,
) -> SpectraCollection | None:
    """Drop spectra with fewer peaks than required."""
    peak_counts = spectrum_in.fragments.count(axis=1)
    thresholds = np.full(len(spectrum_in), n_required, dtype=np.int64)

    if ratio_required is not None and "parent_mass" in spectrum_in.metadata.columns:
        parent_mass = pd.to_numeric(
            spectrum_in.metadata["parent_mass"],
            errors="coerce",
        )

        has_parent_mass = ~np.isnan(parent_mass) & (parent_mass != 0)

        thresholds_by_mass = np.ceil(parent_mass[has_parent_mass] * ratio_required).astype(np.int64)
        thresholds[has_parent_mass] = np.maximum(
            thresholds[has_parent_mass],
            thresholds_by_mass,
        )

    keep_mask = peak_counts >= thresholds

    if not keep_mask.any():
        return None

    return spectrum_in.filter(keep_mask, inplace=not clone)


require_minimum_number_of_peaks = collection_filter(
    _require_minimum_number_of_peaks_spectrum,
    collection_impl=_require_minimum_number_of_peaks_collection,
)
