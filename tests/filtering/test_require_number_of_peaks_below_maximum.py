import numpy as np
from matchms.filtering.peak_processing.require_number_of_peaks_below_maximum import \
    require_number_of_peaks_below_maximum
from matchms.Spectrum import Spectrum


def test_require_number_of_peaks_below_maximum():
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum = Spectrum(mz, intensities)
    assert require_number_of_peaks_below_maximum(spectrum, 3) is None


def test_require_number_of_peaks_below_maximum_not_removed():
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum = Spectrum(mz, intensities)
    assert require_number_of_peaks_below_maximum(spectrum, 10) == spectrum

