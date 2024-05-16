import numpy as np
from matchms.Spectrum import Spectrum
import logging

logger = logging.getLogger("matchms")


def estimate_peak_list_type(spectrum: Spectrum):
    """
    Reproduced from MZmine.
    https://github.com/mzmine/mzmine3/blob/master/src/main/java/io/github/mzmine/util/scans/ScanUtils.java#L609
    """
    peaks_n = spectrum.mz.shape[0]
    if peaks_n < 5:
        return spectrum

    bp_min_i, bp_max_i = _get_peak_intens_neighbourhood(spectrum.intensities)

    bp_span = bp_max_i - bp_min_i + 1
    bp_mz_span = spectrum.mz[bp_max_i] - spectrum.mz[bp_min_i]
    mz_span = spectrum.mz[-1] - spectrum.mz[0]
    if bp_span < 3 or bp_mz_span > mz_span / 1000:
        return spectrum
    logger.info(
        "Spectrum removed because likely profile data. The base peak mz span is %s, "
        "the numbef of fragements in this span is %s and the total mz span is %s",
        bp_mz_span, bp_span, mz_span)
    return None


def _get_peak_intens_neighbourhood(intensities):
    """
    Returns the range of indices around the highest peak that are more than half the intensity of the highest peak.
    """
    base_peak_i = intensities.argmax()
    base_peak_intensity = intensities[base_peak_i]
    intensity_threshold = base_peak_intensity / 2

    # Select true for each peak above threshold. Adding False on both ends, to always get a result from np.argmin.
    threshold_mask = np.concatenate([[False], intensities > intensity_threshold, [False]])

    nr_of_peaks_above_threshold_before_base_peak = np.argmin(np.flip(threshold_mask[:base_peak_i + 1]))
    nr_of_peaks_above_threshold_after_base_peak = np.argmin(threshold_mask[base_peak_i + 2:])

    base_peak_min_i = base_peak_i - nr_of_peaks_above_threshold_before_base_peak
    base_peak_max_i = base_peak_i + nr_of_peaks_above_threshold_after_base_peak
    return base_peak_min_i, base_peak_max_i
