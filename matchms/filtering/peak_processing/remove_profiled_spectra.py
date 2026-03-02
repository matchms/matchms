import logging
from typing import Optional
import numpy as np
from matchms.Spectrum import Spectrum
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def remove_profiled_spectra(
    spectrum_in: Spectrum, mz_window=0.5, clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """Remove profiled spectra

    Spectra are removed if within the mz_window of 0.5 of the highest peak at least 2 peaks next to the main peak are of
    intensity > max_intensity/2.

    Reproduced from MZmine.
    https://github.com/mzmine/mzmine3/blob/master/src/main/java/io/github/mzmine/util/scans/ScanUtils.java#L609

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    mz_window:
        Window of mz values (in Da) that are allowed to lie within
        the top k peaks. Default is 50 Da.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        None if the spectrum is likely profile data, else the input spectrum.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    peaks_n = spectrum.mz.shape[0]
    if peaks_n < 3:
        return spectrum

    number_of_high_intensity_surounding_peaks = _get_number_of_high_intensity_surounding_peaks(
        spectrum.intensities, spectrum.mz, mz_window
    )
    if number_of_high_intensity_surounding_peaks < 3:
        return spectrum
    logger.info(
        "Spectrum removed because likely profile data."
        "Number of high intensity fragments next to the highest peak = %s.",
        number_of_high_intensity_surounding_peaks,
    )
    return None


def _get_number_of_high_intensity_surounding_peaks(intensities, mz, mz_window):
    base_peak_i = intensities.argmax()

    intensities_within_mz_window = _select_intensities_within_mz_window(intensities, mz, mz_window)
    nr_of_peaks_above_threshold_before_base_peak, nr_of_peaks_above_threshold_after_base_peak = (
        _get_peak_intens_neighbourhood(intensities_within_mz_window)
    )
    base_peak_min_i = base_peak_i - nr_of_peaks_above_threshold_before_base_peak
    base_peak_max_i = base_peak_i + nr_of_peaks_above_threshold_after_base_peak

    number_of_high_intensity_surounding_peaks = base_peak_max_i - base_peak_min_i + 1
    return number_of_high_intensity_surounding_peaks


def _select_intensities_within_mz_window(intensities, mz, mz_span):
    base_peak_i = intensities.argmax()
    within_mz_window = (mz > mz[base_peak_i] - mz_span) & (mz < mz[base_peak_i] + mz_span)
    intensities_within_mz_window = intensities[within_mz_window]
    return intensities_within_mz_window


def _get_peak_intens_neighbourhood(intensities):
    """
    Returns the range of indices around the highest peak that are more than half the intensity of the highest peak.
    """

    base_peak_i = intensities.argmax()
    intensity_threshold = intensities[base_peak_i] / 2

    # Select true for each peak above threshold. Adding False on both ends, to always get a result from np.argmin.
    threshold_mask = np.concatenate([[False], intensities > intensity_threshold, [False]])

    nr_of_peaks_above_threshold_before_base_peak = np.argmin(np.flip(threshold_mask[: base_peak_i + 1]))
    nr_of_peaks_above_threshold_after_base_peak = np.argmin(threshold_mask[base_peak_i + 2 :])
    return nr_of_peaks_above_threshold_before_base_peak, nr_of_peaks_above_threshold_after_base_peak
