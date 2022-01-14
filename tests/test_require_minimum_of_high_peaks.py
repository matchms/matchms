import numpy
import pytest
from matchms.filtering import require_minimum_of_high_peaks
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("peaks, no_peaks, intensity_percent, expected", [
    [
        [numpy.array([10, 20, 30, 40], dtype="float"), numpy.array(
            [0, 1, 10, 100], dtype="float")], 2, 2,
        SpectrumBuilder().with_mz(numpy.array([10, 20, 30, 40], dtype="float")).with_intensities(
            numpy.array([0, 1, 10, 100], dtype="float")).build()
    ], [
        [numpy.array([10, 20, 30, 40], dtype="float"), numpy.array(
            [0, 1, 10, 100], dtype="float")], 5, 2, None
    ], [
        [numpy.array([10, 20, 30, 40, 50, 60, 70], dtype="float"),
         numpy.array([0, 1, 10, 25, 50, 75, 100], dtype="float")],
        2, 10, SpectrumBuilder().with_mz(numpy.array([10, 20, 30, 40, 50, 60, 70], dtype="float")).with_intensities(
            numpy.array([0, 1, 10, 25, 50, 75, 100], dtype="float")).build()
    ]
])
def test_require_mininum_of_high_peaks(peaks, no_peaks, intensity_percent, expected):
    spectrum_in = SpectrumBuilder().with_mz(
        peaks[0]).with_intensities(peaks[1]).build()

    spectrum = require_minimum_of_high_peaks(
        spectrum_in, no_peaks=no_peaks, intensity_percent=intensity_percent)

    assert spectrum == expected


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = require_minimum_of_high_peaks(spectrum_in, no_peaks=2)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), "Expected input spectrum to remain unchanged."


def test_with_input_none():
    """Test if input spectrum is None."""
    spectrum_in = None
    spectrum = require_minimum_of_high_peaks(spectrum_in)
    assert spectrum is None
