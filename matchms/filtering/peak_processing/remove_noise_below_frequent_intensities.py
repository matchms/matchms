import logging
import numpy as np
from scipy.sparse import csr_array
from matchms.filtering._dispatch import collection_filter
from matchms.fragment_collection import CSRFragmentCollection
from matchms.fragments import Fragments
from matchms.spectra_collection import SpectraCollection
from matchms.spectrum import Spectrum
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _remove_noise_below_frequent_intensities(
    spectrum_in: Spectrum,
    min_count_of_frequent_intensities: int = 5,
    noise_level_multiplier: float = 2.0,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Remove noise inferred from frequently repeated intensity values.

    Spectra without prior noise filtering can contain many peaks with exactly
    repeated intensity values. Of all intensity values occurring at least
    ``min_count_of_frequent_intensities`` times, the highest is selected.
    The noise threshold is this intensity multiplied by
    ``noise_level_multiplier``. Only peaks strictly above that threshold are retained.

    If no intensity occurs frequently enough, the spectrum is returned unchanged.

    This filter was suggested by Tytus Mak.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    min_count_of_frequent_intensities:
        Minimum number of occurrences required for an intensity value to be
        considered frequent.
    noise_level_multiplier:
        From all intensities that repeat more than min_count_of_frequent_intensities the highest is selected.
        The noise level is set to this intensity * noise_level_multiplier.
    clone:
        If True, return a filtered copy. If False, modify the input spectrum.

    Returns
    -------
    Spectrum or None
        Filtered spectrum, or None if the input is None.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    highest_frequent_peak = _select_highest_frequent_peak(
        spectrum.intensities,
        min_count_of_frequent_intensities,
    )

    if highest_frequent_peak == -1:
        return spectrum

    noise_threshold = (
        highest_frequent_peak
        * noise_level_multiplier
    )

    peaks_to_keep = (
        spectrum.intensities > noise_threshold
    )

    spectrum.peaks = Fragments(
        mz=spectrum.mz[peaks_to_keep],
        intensities=spectrum.intensities[peaks_to_keep],
    )

    logger.info(
        "Fragments removed with intensity below %s",
        noise_threshold,
    )

    return spectrum


def _remove_noise_below_frequent_intensities_collection(
    spectrum_in: SpectraCollection,
    min_count_of_frequent_intensities: int = 5,
    noise_level_multiplier: float = 2.0,
    clone: bool | None = True,
) -> SpectraCollection:
    """Apply frequent-intensity noise removal directly to CSR fragments."""
    if not isinstance(
        spectrum_in.fragments,
        CSRFragmentCollection,
    ):
        raise NotImplementedError(
            "Native SpectraCollection processing for "
            "remove_noise_below_frequent_intensities currently supports only "
            "CSRFragmentCollection."
        )

    target = spectrum_in.copy() if clone else spectrum_in

    fragments = target.fragments
    array = fragments.array

    if array.nnz == 0:
        return target

    keep = np.ones(array.data.size, dtype=bool)
    n_filtered_spectra = 0

    for row in range(len(target)):
        start = array.indptr[row]
        end = array.indptr[row + 1]

        intensities = array.data[start:end]

        # A repeated value cannot reach the requested count when the complete
        # spectrum contains fewer peaks than that count.
        if (
            intensities.size
            < min_count_of_frequent_intensities
        ):
            continue

        highest_frequent_peak = _select_highest_frequent_peak(
            intensities,
            min_count_of_frequent_intensities,
        )

        if highest_frequent_peak == -1:
            continue

        noise_threshold = (
            highest_frequent_peak
            * noise_level_multiplier
        )

        row_keep = intensities > noise_threshold

        if not np.all(row_keep):
            keep[start:end] = row_keep
            n_filtered_spectra += 1

    if n_filtered_spectra == 0:
        return target

    # CSR entries are stored consecutively by spectrum. The cumulative number
    # of retained entries at each original row boundary therefore gives the
    # indptr array of the filtered CSR matrix.
    cumulative_kept = np.empty(
        keep.size + 1,
        dtype=np.int64,
    )
    cumulative_kept[0] = 0

    np.cumsum(
        keep,
        dtype=np.int64,
        out=cumulative_kept[1:],
    )

    new_indptr = cumulative_kept[array.indptr]

    filtered_array = csr_array(
        (
            array.data[keep],
            array.indices[keep],
            new_indptr,
        ),
        shape=array.shape,
    )

    target._fragments = CSRFragmentCollection.from_array(
        filtered_array,
        mz_precision=fragments.mz_precision,
        mz_rounding=fragments.mz_rounding,
        index_dtype=fragments.index_dtype,
    )

    target._clear_cache()

    logger.info(
        "Removed frequent-intensity noise from %d spectra.",
        n_filtered_spectra,
    )

    return target


def _select_highest_frequent_peak(
    intensities,
    min_count_of_frequent_intensities=5,
):
    """Return the highest intensity occurring at least the requested count."""
    unique_values, counts = np.unique(
        intensities,
        return_counts=True,
    )

    mask = (
        counts
        >= min_count_of_frequent_intensities
    )
    filtered_values = unique_values[mask]

    if filtered_values.size > 0:
        return filtered_values.max()

    return -1


remove_noise_below_frequent_intensities = collection_filter(
    _remove_noise_below_frequent_intensities,
    collection_impl=(
        _remove_noise_below_frequent_intensities_collection
    ),
)