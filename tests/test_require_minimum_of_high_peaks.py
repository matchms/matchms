import numpy
from matchms import Spectrum
from matchms.filtering import require_minimum_of_high_peaks


def test_require_minimum_of_high_peaks_no_params():
    """No parameters provided."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_minimum_of_high_peaks(spectrum_in)

    assert spectrum is None, "Expected None as the number of peaks (4) is less than the default of 5 for no_peaks."


def test_require_minimum_of_high_peaks_no_peaks_2():
    """Set no_peaks to 2."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_minimum_of_high_peaks(spectrum_in, no_peaks=2)

    assert spectrum == spectrum_in, "Expected no changes."


def test_require_minimum_of_high_peaks_intensity_percent_10():
    """Set intensity_percent to 10."""
    mz = numpy.array([10, 20, 30, 40, 50, 60, 70], dtype="float")
    intensities = numpy.array([0, 1, 10, 25, 50, 75, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_minimum_of_high_peaks(spectrum_in, intensity_percent=10)

    assert spectrum == spectrum_in, "Expected no changes."


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_minimum_of_high_peaks(spectrum_in, no_peaks=2)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), "Expected input spectrum to remain unchanged."


def test_with_input_none():
    """Test if input spectrum is None."""
    spectrum_in = None
    spectrum = require_minimum_of_high_peaks(spectrum_in)
    assert spectrum is None
