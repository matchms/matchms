import numpy as np
from matchms.filtering._dispatch import collection_filter
from matchms.fragments import Fragments
from matchms.spectra_collection import SpectraCollection
from matchms.typing import SpectrumType


def _validate_offset_to_precursor(offset_to_precursor) -> float:
    """Validate and convert the precursor offset to float."""
    if not isinstance(
        offset_to_precursor,
        (float, int, np.floating, np.integer),
    ):
        raise TypeError(
            "Expected 'offset_to_precursor' to be a scalar number."
        )

    return float(offset_to_precursor)


def _remove_peaks_relative_to_precursor_mz(
    spectrum_in: SpectrumType,
    offset_to_precursor: float = -1.6,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Remove peaks above ``precursor_mz + offset_to_precursor``.

    Peaks with m/z values larger than the precursor m/z plus the specified
    offset are removed. With the default negative offset, peaks close to and
    above the precursor ion are removed.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    offset_to_precursor:
        All peaks with mz values > precursor_mz + offset_to_precursor will be removed.
        Default is -1.6 Da based Flash Entropy article by Li and Fiehn, 2023, Nature Methods.
        (see https://www.nature.com/articles/s41592-023-02012-9)
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

    precursor_mz = spectrum.get("precursor_mz", None)
    if precursor_mz is None:
        raise TypeError("Undefined 'precursor_mz'.")

    if not isinstance(precursor_mz, (float, int)):
        raise TypeError(
            "Expected 'precursor_mz' to be a scalar number. "
            "Consider applying 'add_precursor_mz' filter first."
        )

    offset_to_precursor = _validate_offset_to_precursor(
        offset_to_precursor
    )

    threshold = float(precursor_mz) + offset_to_precursor

    mz = spectrum.peaks.mz
    intensities = spectrum.peaks.intensities
    keep = mz <= threshold

    spectrum.peaks = Fragments(
        mz=mz[keep],
        intensities=intensities[keep],
    )

    return spectrum


def _remove_peaks_relative_to_precursor_mz_collection(
    spectrum_in: SpectraCollection,
    offset_to_precursor: float = -1.6,
    clone: bool | None = True,
) -> SpectraCollection:
    """Remove precursor-region peaks directly from a SpectraCollection."""
    offset_to_precursor = _validate_offset_to_precursor(
        offset_to_precursor
    )

    metadata = spectrum_in.metadata

    if "precursor_mz" not in metadata.columns:
        raise TypeError("Undefined 'precursor_mz'.")

    try:
        precursor_mz = metadata["precursor_mz"].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Expected 'precursor_mz' to be a scalar. "
            "Consider applying 'add_precursor_mz' filter first."
        ) from exc

    if np.isnan(precursor_mz).any():
        raise TypeError(
            "Undefined 'precursor_mz'. "
            "Consider applying 'add_precursor_mz' filter first."
        )

    thresholds = precursor_mz + offset_to_precursor

    target = spectrum_in.copy() if clone else spectrum_in

    target._fragments = (
        target.fragments.select_by_mz_upper_bound_per_row(
            thresholds
        )
    )
    target._clear_cache()

    return target


remove_peaks_relative_to_precursor_mz = collection_filter(
    _remove_peaks_relative_to_precursor_mz,
    collection_impl=_remove_peaks_relative_to_precursor_mz_collection,
)