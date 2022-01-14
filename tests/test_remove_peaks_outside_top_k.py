import numpy
import pytest
from matchms.filtering import remove_peaks_outside_top_k
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("peaks, k, mz_window, expected", [
    [
        [numpy.array([1, 25, 60, 70, 80, 90, 100, 110], dtype="float"), numpy.array(
            [10, 25, 50, 75, 100, 125, 150, 175], dtype="float")],
        6, 50, [25., 60., 70., 80., 90., 100., 110.]
    ], [
        [numpy.array([1, 25, 60, 70, 80], dtype="float"), numpy.array(
            [10, 25, 50, 75, 100], dtype="float")], 6, 50, numpy.array([1, 25, 60, 70, 80], dtype="float")
    ], [
        [numpy.array([1, 25, 60, 70, 80, 90], dtype="float"), numpy.array(
            [10, 25, 50, 75, 100, 125], dtype="float")], 6, 50, numpy.array([1, 25, 60, 70, 80, 90], dtype="float")
    ], [
        [numpy.array([1, 20, 30, 50, 55, 90, 100, 110], dtype="float"), numpy.array(
            [10, 25, 50, 75, 100, 125, 150, 175], dtype="float")], 3, 50, [50., 55., 90., 100., 110.]
    ], [
        [numpy.array([1, 10, 30, 40, 50, 60, 70, 80], dtype="float"), numpy.array(
            [10, 25, 50, 75, 100, 125, 150, 175], dtype="float")], 6, 10, [30., 40., 50., 60., 70., 80.]
    ]
])
def test_remove_peaks_outside_top_k(peaks, k, mz_window, expected):
    spectrum_in = SpectrumBuilder().with_mz(
        peaks[0]).with_intensities(peaks[1]).build()

    spectrum = remove_peaks_outside_top_k(
        spectrum_in, k=k, mz_window=mz_window)

    num_peaks = len(expected)
    assert len(
        spectrum.peaks) == num_peaks, "Expected ${num_peaks} peaks to remain."
    numpy.testing.assert_array_equal(
        spectrum.peaks.mz, expected, err_msg="Expected different peaks to remain.")


def test_remove_peaks_outside_top_k_no_params_check_sorting():
    """Check sorting of mzs and intensities with default parameters."""
    mz = numpy.array([100, 270, 300, 400, 500, 600, 700, 710, 720, 1000], dtype="float")
    intensities = numpy.array([100, 200, 50, 175, 10, 125, 150, 75, 25, 1], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = remove_peaks_outside_top_k(spectrum_in)

    assert len(spectrum.peaks) == 8, "Expected 8 peaks to remain."
    assert spectrum.peaks.mz.tolist() == [100., 270., 300., 400., 600., 700., 710., 720.], "Expected different mzs to remain."
    assert spectrum.peaks.intensities.tolist() == [100., 200., 50., 175., 125., 150., 75., 25.], "Expected different intensities to remain."


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    spectrum_in = SpectrumBuilder().build()

    spectrum = remove_peaks_outside_top_k(spectrum_in)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), "Expected input spectrum to remain unchanged."


def test_with_input_none():
    """Test if input spectrum is None."""
    spectrum_in = None
    spectrum = remove_peaks_outside_top_k(spectrum_in)
    assert spectrum is None
