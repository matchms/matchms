import logging
import numpy as np
from matchms.filtering._dispatch import collection_filter
from matchms.fragment_collection import CSRFragmentCollection
from matchms.spectra_collection import SpectraCollection
from matchms.spectrum import Spectrum
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _remove_profiled_spectra(
    spectrum_in: Spectrum,
    mz_window: float = 0.5,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Remove spectra that are likely profile-mode data.

    A spectrum is considered likely profile data when the base peak belongs to
    a contiguous group of at least three peaks that

    - lie within ``mz_window`` Da of the base peak, and
    - have intensities greater than half the base-peak intensity.

    This criterion is reproduced from MZmine.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    mz_window:
        Maximum m/z distance in Da from the base peak considered when checking
        neighboring high-intensity peaks. Default is 0.5 Da.
    clone:
        If True, clone spectra that are retained. If False, return the original
        spectrum unchanged.

    Returns
    -------
    Spectrum or None
        ``None`` if the spectrum is likely profile data; otherwise the retained
        spectrum.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if _is_profiled_spectrum(
        spectrum.intensities,
        spectrum.mz,
        mz_window,
    ):
        logger.info(
            "Spectrum removed because it is likely profile data."
        )
        return None

    return spectrum


def _remove_profiled_spectra_collection(
    spectrum_in: SpectraCollection,
    mz_window: float = 0.5,
    clone: bool | None = True,
) -> SpectraCollection | None:
    """Remove likely profile spectra directly from CSR fragment storage."""
    if not isinstance(
        spectrum_in.fragments,
        CSRFragmentCollection,
    ):
        raise NotImplementedError(
            "Native SpectraCollection processing for remove_profiled_spectra "
            "currently supports only CSRFragmentCollection."
        )

    fragments = spectrum_in.fragments
    array = fragments.array

    keep_mask = np.ones(
        len(spectrum_in),
        dtype=bool,
    )

    for row in range(len(spectrum_in)):
        start = array.indptr[row]
        end = array.indptr[row + 1]

        if end - start < 3:
            continue

        intensities = array.data[start:end]
        mz = fragments.bin_to_mz(
            array.indices[start:end]
        )

        if _is_profiled_spectrum(
            intensities,
            mz,
            mz_window,
        ):
            keep_mask[row] = False

    n_removed = int((~keep_mask).sum())

    if n_removed == 0:
        return (
            spectrum_in.copy()
            if clone
            else spectrum_in
        )

    logger.info(
        "Removed %d spectra because they are likely profile data.",
        n_removed,
    )

    # Preserving the behavior of the former spectrum-wise handling:
    # if all spectra are removed, return None.
    if not np.any(keep_mask):
        return None

    keep_indices = np.flatnonzero(keep_mask)

    if not clone:
        spectrum_in._reorder(keep_indices)
        return spectrum_in

    filtered_fragments = fragments.take(
        keep_indices
    )
    filtered_metadata = (
        spectrum_in._metadata
        .iloc[keep_indices]
        .reset_index(drop=True)
    )

    return spectrum_in.__class__._from_metadata_and_fragments(
        metadata=filtered_metadata,
        fragments=filtered_fragments,
        mz_precision=spectrum_in.mz_precision,
    )


def _is_profiled_spectrum(
    intensities: np.ndarray,
    mz: np.ndarray,
    mz_window: float,
) -> bool:
    """Return whether a peak pattern matches the profile-data criterion."""
    if mz.size < 3:
        return False

    number_of_surrounding_peaks = (
        _get_number_of_high_intensity_surrounding_peaks(
            intensities,
            mz,
            mz_window,
        )
    )

    return number_of_surrounding_peaks >= 3


def _get_number_of_high_intensity_surrounding_peaks(
    intensities: np.ndarray,
    mz: np.ndarray,
    mz_window: float,
) -> int:
    """Count contiguous high-intensity peaks around the base peak."""
    intensities_within_mz_window = (
        _select_intensities_within_mz_window(
            intensities,
            mz,
            mz_window,
        )
    )

    (
        n_before,
        n_after,
    ) = _get_peak_intensity_neighbourhood(
        intensities_within_mz_window
    )

    # The neighboring-peak counts exclude the base peak itself.
    return int(n_before + n_after + 1)


def _select_intensities_within_mz_window(
    intensities: np.ndarray,
    mz: np.ndarray,
    mz_window: float,
) -> np.ndarray:
    """Return intensities strictly within the m/z window around the base peak."""
    base_peak_index = int(
        intensities.argmax()
    )
    base_peak_mz = mz[base_peak_index]

    within_mz_window = (
        (mz > base_peak_mz - mz_window)
        & (mz < base_peak_mz + mz_window)
    )

    return intensities[within_mz_window]


def _get_peak_intensity_neighbourhood(
    intensities: np.ndarray,
) -> tuple[int, int]:
    """Count contiguous >50% intensity neighbors around the base peak."""
    base_peak_index = int(
        intensities.argmax()
    )
    intensity_threshold = (
        intensities[base_peak_index] / 2
    )

    # Add False at both boundaries so the first below-threshold position always
    # exists when searching outward from the base peak.
    threshold_mask = np.concatenate([[False], intensities > intensity_threshold, [False]])

    n_before = np.argmin(
        np.flip(
            threshold_mask[
                : base_peak_index + 1
            ]
        )
    )

    n_after = np.argmin(
        threshold_mask[
            base_peak_index + 2 :
        ]
    )

    return int(n_before), int(n_after)


remove_profiled_spectra = collection_filter(
    _remove_profiled_spectra,
    collection_impl=_remove_profiled_spectra_collection,
)