import numpy
from matchms import Spectrum
from matchms.filtering import remove_peaks_outside_top_k


def test_remove_peaks_outside_top_k_no_params():
    """No parameters provided."""
    mz = numpy.array([1, 25, 60, 70, 80, 90, 100, 110], dtype="float")
    intensities = numpy.array([10, 25, 50, 75, 100, 125, 150, 175], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = remove_peaks_outside_top_k(spectrum_in)

    assert len(spectrum.peaks) == 7, "Expected 7 peaks to remain."
    assert spectrum.peaks.mz.tolist() == [25., 60., 70., 80., 90., 100., 110.], "Expected different peaks to remain."


def test_remove_peaks_outside_top_k_no_params_peaks_less_than_k():
    """Check case when the number of peaks is less than k with default parameters."""
    mz = numpy.array([1, 25, 60, 70, 80], dtype="float")
    intensities = numpy.array([10, 25, 50, 75, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = remove_peaks_outside_top_k(spectrum_in)

    assert spectrum == spectrum_in, "Expected no changes as the number of peaks (5) is less than k (6)."


def test_remove_peaks_outside_top_k_no_params_peaks_equal_to_k():
    """Check case when the number of peaks is equal to k with default parameters."""
    mz = numpy.array([1, 25, 60, 70, 80, 90], dtype="float")
    intensities = numpy.array([10, 25, 50, 75, 100, 125], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = remove_peaks_outside_top_k(spectrum_in)

    assert spectrum == spectrum_in, "Expected no changes as the number of peaks (6) is equal to k (6)."


def test_remove_peaks_outside_top_k_no_params_check_sorting():
    """Check sorting of mzs and intensities with default parameters."""
    mz = numpy.array([100, 270, 300, 400, 500, 600, 700, 710, 720, 1000], dtype="float")
    intensities = numpy.array([100, 200, 50, 175, 10, 125, 150, 75, 25, 1], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = remove_peaks_outside_top_k(spectrum_in)

    assert len(spectrum.peaks) == 8, "Expected 8 peaks to remain."
    assert spectrum.peaks.mz.tolist() == [100., 270., 300., 400., 600., 700., 710., 720.], "Expected different mzs to remain."
    assert spectrum.peaks.intensities.tolist() == [100., 200., 50., 175., 125., 150., 75., 25.], "Expected different intensities to remain."


def test_remove_peaks_outside_top_k_3():
    """Set k to 3."""
    mz = numpy.array([1, 20, 30, 50, 55, 90, 100, 110], dtype="float")
    intensities = numpy.array([10, 25, 50, 75, 100, 125, 150, 175], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = remove_peaks_outside_top_k(spectrum_in, k=3)

    assert len(spectrum.peaks) == 5, "Expected 5 peaks to remain."
    assert spectrum.peaks.mz.tolist() == [50., 55., 90., 100., 110.], "Expected different peaks to remain."


def test_remove_peaks_outside_top_k_mz_window_10():
    """Set mz_window to 10."""
    mz = numpy.array([1, 10, 30, 40, 50, 60, 70, 80], dtype="float")
    intensities = numpy.array([10, 25, 50, 75, 100, 125, 150, 175], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = remove_peaks_outside_top_k(spectrum_in, mz_window=10)

    assert len(spectrum.peaks) == 6, "Expected 6 peaks to remain."
    assert spectrum.peaks.mz.tolist() == [30., 40., 50., 60., 70., 80.], "Expected different peaks to remain."
