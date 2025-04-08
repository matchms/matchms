import numpy as np
import pytest
from matchms.filtering.peak_processing.remove_profiled_spectra import _get_number_of_high_intensity_surounding_peaks, _get_peak_intens_neighbourhood


@pytest.mark.parametrize(
    "intensities, expected_no_peaks_before, expected_no_peaks_after",
    [
        [[0.7, 0.6, 0.8, 1.0, 0.9], 3, 1],  # All peaks included
        [[0.3, 0.6, 0.8, 1.0, 0.9], 2, 1],
        [[0.7, 0.6, 0.8, 1.0, 0.2], 3, 0],
        [[0.3, 0.6, 0.8, 1.0, 0.2], 2, 0],
        [[0.3, 0.6, 0.8, 1.0, 0.2, 0.8, 0.9, 0.1], 2, 0],
        [[0.8, 0.3, 0.6, 0.8, 1.0, 0.2, 0.8, 0.9, 0.1], 2, 0],
    ],
)
def test_get_peak_intens_neighbourhood(intensities, expected_no_peaks_before, expected_no_peaks_after):
    peaks_before, peaks_after = _get_peak_intens_neighbourhood(np.array(intensities))
    assert peaks_before == expected_no_peaks_before
    assert peaks_after == expected_no_peaks_after


@pytest.mark.parametrize(
    "intensities, mz, expected_no_peaks",
    [
        [[0.7, 0.6, 0.8, 1.0, 0.9], [0.1, 0.2, 0.3, 0.4, 0.5], 5],  # All peaks included
        [[0.3, 0.6, 0.8, 1.0, 0.9], [0.1, 0.2, 0.3, 0.4, 0.5], 4],
        [[0.7, 0.6, 0.8, 1.0, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5], 4],
        [[0.3, 0.6, 0.8, 1.0, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5], 3],
        [[0.3, 0.6, 0.8, 1.0, 0.2, 0.8, 0.9, 0.1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 3],
        [[0.8, 0.3, 0.6, 0.8, 1.0, 0.2, 0.8, 0.9, 0.1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 3],
        [[0.7, 0.6, 0.8, 1.0, 0.9], [10.0, 20.0, 20.1, 20.2, 20.3], 4],
    ],
)
def test_get_number_of_high_intensity_surounding_peaks(intensities, mz, expected_no_peaks):
    number_of_high_intensity_surounding_peaks = _get_number_of_high_intensity_surounding_peaks(np.array(intensities), mz=np.array(mz), mz_window=1.0)
    assert number_of_high_intensity_surounding_peaks == expected_no_peaks
