import logging
from numbers import Real
import numpy as np
from scipy.sparse import csr_array
from tqdm.auto import tqdm
from matchms.filtering._dispatch import collection_filter
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType
from ...Fragments import Fragments


logger = logging.getLogger("matchms")


def _validate_scale_to_max(scale_to_max: float) -> float:
    """Validate and normalize the scale_to_max argument."""
    if not isinstance(scale_to_max, Real):
        raise TypeError("'scale_to_max' must be a positive number.")

    scale_to_max = float(scale_to_max)

    if scale_to_max <= 0:
        raise ValueError("'scale_to_max' must be > 0.")

    return scale_to_max


def _normalize_intensities_spectrum(
    spectrum_in: SpectrumType,
    clone: bool | None = True,
    scale_to_max: float = 1.0,
) -> SpectrumType | None:
    """Normalize peak intensities relative to the maximum peak intensity.

    Intensities are divided by the maximum intensity of the spectrum and then
    multiplied by ``scale_to_max``. By default, this normalizes spectra to unit
    height, i.e. the most intense peak receives intensity ``1.0``.

    Peaks with zero intensity are removed. Negative peak intensities are not
    allowed and raise a ``ValueError``.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.
    scale_to_max:
        Desired intensity of the most intense peak after normalization.
        Default is ``1.0``. For example, ``scale_to_max=1000.0`` scales the
        base peak to intensity 1000.

    Returns
    -------
    Spectrum or None
        Spectrum with normalized intensities, or ``None`` if input is ``None``.
    """
    if spectrum_in is None:
        return None

    scale_to_max = _validate_scale_to_max(scale_to_max)

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if len(spectrum.peaks) == 0:
        return spectrum

    mz = spectrum.peaks.mz
    intensities = spectrum.peaks.intensities

    if np.any(intensities < 0):
        raise ValueError("Negative peak intensities are not allowed.")

    keep = intensities > 0
    mz = mz[keep]
    intensities = intensities[keep]

    if intensities.size == 0:
        logger.warning("Peaks of spectrum with all peak intensities <= 0 were deleted.")
        spectrum.peaks = Fragments(mz=np.array([]), intensities=np.array([]))
        return spectrum

    max_intensity = np.max(intensities)
    normalized_intensities = intensities / max_intensity * scale_to_max

    spectrum.peaks = Fragments(mz=mz, intensities=normalized_intensities)
    return spectrum


def _normalize_intensities_collection(
    spectrum_in: SpectraCollection,
    clone: bool | None = True,
    scale_to_max: float = 1.0,
    progress_bar: bool = False,
) -> SpectraCollection:
    """Normalize intensities row-wise for a SpectraCollection."""
    scale_to_max = _validate_scale_to_max(scale_to_max)

    target = spectrum_in.copy() if clone else spectrum_in

    fragments = target.fragments
    array = fragments.array.copy().tocsr()

    if array.nnz == 0:
        return target

    if np.any(array.data < 0):
        raise ValueError("Negative peak intensities are not allowed.")

    indptr = array.indptr
    indices = array.indices

    # Use float dtype to avoid integer truncation if the sparse array was
    # constructed from integer intensities.
    data = array.data.astype(float, copy=True)

    new_data_parts = []
    new_indices_parts = []
    new_indptr = [0]

    for row_idx in tqdm(range(array.shape[0]), disable=not progress_bar):
        start, end = indptr[row_idx], indptr[row_idx + 1]

        if start == end:
            new_indptr.append(new_indptr[-1])
            continue

        row_data = data[start:end]
        row_indices = indices[start:end]

        keep = row_data > 0
        row_data = row_data[keep]
        row_indices = row_indices[keep]

        if row_data.size == 0:
            logger.warning("Peaks of spectrum with all peak intensities <= 0 were deleted.")
            new_indptr.append(new_indptr[-1])
            continue

        max_intensity = np.max(row_data)
        normalized = row_data / max_intensity * scale_to_max

        new_data_parts.append(normalized)
        new_indices_parts.append(row_indices)
        new_indptr.append(new_indptr[-1] + normalized.size)

    if new_data_parts:
        new_data = np.concatenate(new_data_parts)
        new_indices = np.concatenate(new_indices_parts).astype(array.indices.dtype, copy=False)
    else:
        new_data = np.array([], dtype=float)
        new_indices = np.array([], dtype=array.indices.dtype)

    normalized_array = csr_array(
        (
            new_data,
            new_indices,
            np.asarray(new_indptr, dtype=array.indptr.dtype),
        ),
        shape=array.shape,
    )

    target._fragments = fragments.__class__.from_array(
        normalized_array,
        bin_size=fragments.bin_size,
    )
    target._clear_cache(["fragment_hashes", "spectra_hashes"])

    return target


normalize_intensities = collection_filter(
    _normalize_intensities_spectrum,
    collection_impl=_normalize_intensities_collection,
)
