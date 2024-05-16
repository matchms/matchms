from matchms.filtering.peak_processing.remove_profiled_spectra import _get_peak_intens_neighbourhood
import numpy as np
import pytest


@pytest.mark.parametrize("intensities, expected_min_peak_i, expected_max_peak_i", [
    [[0.7, 0.6, 0.8, 1.0, 0.9], 0, 4], # All peaks included
    [[0.3, 0.6, 0.8, 1.0, 0.9], 1, 4],
    [[0.7, 0.6, 0.8, 1.0, 0.2], 0, 3],
    [[0.3, 0.6, 0.8, 1.0, 0.2], 1, 3],
    [[0.3, 0.6, 0.8, 1.0, 0.2, 0.8, 0.9, 0.1], 1, 3],
    [[0.8, 0.3, 0.6, 0.8, 1.0, 0.2, 0.8, 0.9, 0.1], 2, 4],
])
def test_get_peak_intens_neighbourhood(intensities, expected_min_peak_i, expected_max_peak_i):
    min_peak_i, max_peak_i = _get_peak_intens_neighbourhood(np.array(intensities))
    assert min_peak_i == expected_min_peak_i
    assert max_peak_i == expected_max_peak_i
