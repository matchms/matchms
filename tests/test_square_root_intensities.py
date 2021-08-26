import numpy
from matchms.filtering import square_root_intensities


def test_square_root_intensities(spectrum_without_losses, mz, intensities):
    """Test if peak intensities are normalized correctly."""

    spectrum = square_root_intensities(spectrum_without_losses)

    expected_intensities = numpy.sqrt(intensities)

    assert numpy.array_equal(spectrum.peaks.intensities, expected_intensities), "Expected different intensities"
    assert numpy.array_equal(spectrum.peaks.mz, mz), "Expected different peak mz."


def test_square_root_intensities_losses_present(spectrum_with_losses, mz, intensities):
    """Test if also losses (if present) are normalized correctly."""

    spectrum = square_root_intensities(spectrum_with_losses)

    expected_intensities = numpy.sqrt(intensities)
    expected_loss_intensities = numpy.sqrt(spectrum_with_losses.losses.intensities)

    assert numpy.array_equal(spectrum.peaks.intensities, expected_intensities), "Expected different intensities"
    assert numpy.array_equal(spectrum.losses.intensities, expected_loss_intensities), "Expected different loss intensities"
    assert numpy.array_equal(spectrum.peaks.mz, mz), "Expected different peak mz."


def test_square_root_intensities_empty_peaks(spectrum_without_peaks):
    """Test running filter with empty peaks spectrum."""

    spectrum = square_root_intensities(spectrum_without_peaks)

    assert spectrum == spectrum_without_peaks, "Spectrum should remain unchanged."


def test_square_root_intensities_empty_spectrum():
    """Test running filter with spectrum == None."""
    spectrum = square_root_intensities(None)

    assert spectrum is None, "Expected spectrum to be None."
