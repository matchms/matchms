import logging
from math import ceil
from typing import Optional
import numpy as np
import pandas as pd
from matchms.filtering._dispatch import collection_filter
from matchms.Fragments import Fragments
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _reduce_to_number_of_peaks_spectrum(
    spectrum_in: SpectrumType,
    n_required: int = 0,
    n_max: int = np.inf,
    ratio_desired: Optional[float] = None,
    clone: Optional[bool] = True,
) -> Optional[SpectrumType]:
    """Lowest intensity peaks will be removed when it has more peaks than desired.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    n_required:
        Number of minimum required peaks. Spectra with fewer peaks will be set
        to 'None'. Default is 1.
    n_max:
        Maximum number of peaks. Remove peaks if more peaks are found. Default is inf.
    ratio_desired:
        Set desired ratio between maximum number of peaks and parent mass.
        For spectra without parent mass (e.g. GCMS spectra) this will raise an
        error when ratio_desired is used.
        Default is None.
    clone:
        Optionally clone the Spectrum.

    """

    def _set_maximum_number_of_peaks_to_keep():
        parent_mass = spectrum.get("parent_mass", None)
        if parent_mass and ratio_desired:
            n_desired_by_mass = int(ceil(ratio_desired * parent_mass))
            return min(max(n_required, n_desired_by_mass), n_max)
        if not ratio_desired:
            return n_max
        raise ValueError("Cannot use ratio_desired for spectrum without parent_mass.")

    def _remove_lowest_intensity_peaks():
        mz, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
        idx = intensities.argsort()[-threshold:]
        idx_sort_by_mz = mz[idx].argsort()
        spectrum.peaks = Fragments(
            mz=mz[idx][idx_sort_by_mz],
            intensities=intensities[idx][idx_sort_by_mz],
        )

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if spectrum.peaks.intensities.size < n_required:
        logger.info(
            "Spectrum with %s (<%s) peaks was set to None.",
            str(spectrum.peaks.intensities.size),
            str(n_required),
        )
        return None

    threshold = _set_maximum_number_of_peaks_to_keep()
    if spectrum.peaks.intensities.size < threshold:
        return spectrum

    _remove_lowest_intensity_peaks()

    return spectrum


def _maximum_number_of_peaks_to_keep_per_row(
    collection: SpectraCollection,
    n_required: int,
    n_max: int | float,
    ratio_desired: float | None,
) -> np.ndarray:
    """Compute row-wise maximum number of peaks to keep."""
    if ratio_desired is None:
        return np.full(len(collection), n_max, dtype=float)

    if "parent_mass" not in collection.metadata.columns:
        raise ValueError("Cannot use ratio_desired for spectrum without parent_mass.")

    parent_mass = pd.to_numeric(
        collection.metadata["parent_mass"],
        errors="coerce",
    )

    if parent_mass.isna().any() or (parent_mass == 0).any():
        raise ValueError("Cannot use ratio_desired for spectrum without parent_mass.")

    desired_by_mass = np.ceil(ratio_desired * parent_mass.to_numpy(dtype=float))
    return np.minimum(
        np.maximum(n_required, desired_by_mass),
        n_max,
    )


def _reduce_to_number_of_peaks_collection(
    collection: SpectraCollection,
    n_required: int = 0,
    n_max: int = np.inf,
    ratio_desired: Optional[float] = None,
    clone: Optional[bool] = True,
    progress_bar: bool = False,
) -> Optional[SpectraCollection]:
    """Collection-native implementation of reduce_to_number_of_peaks."""
    peak_counts = collection.fragments.count(axis=1)

    keep_rows = peak_counts >= n_required
    if not np.any(keep_rows):
        logger.info(
            "All spectra had fewer than %s peaks and were removed.",
            str(n_required),
        )
        return None

    target = collection.copy() if clone else collection

    if not np.all(keep_rows):
        target = target.filter(keep_rows, inplace=False)

    k_per_row = _maximum_number_of_peaks_to_keep_per_row(
        target,
        n_required=n_required,
        n_max=n_max,
        ratio_desired=ratio_desired,
    )

    peak_counts = target.fragments.count(axis=1)
    k_per_row = np.minimum(k_per_row, peak_counts).astype(int)

    # Rows with fewer than k peaks are unchanged by the backend.
    target._fragments = target.fragments.keep_top_k_per_row_variable(
        k_per_row, progress_bar=progress_bar
    )
    target._clear_cache(["fragment_hashes", "spectra_hashes"])

    return target


reduce_to_number_of_peaks = collection_filter(
    _reduce_to_number_of_peaks_spectrum,
    collection_impl=_reduce_to_number_of_peaks_collection,
)
