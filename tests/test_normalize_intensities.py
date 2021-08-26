import numpy
from matchms.filtering import normalize_intensities


def test_square_root_intensities(spectrum_without_losses, mz, intensities):
    """Test if peak intensities are square root correctly."""

    spectrum = normalize_intensities(spectrum_without_losses)

    assert max(spectrum.peaks.intensities) == 1.0, "Expected the spectrum to be scaled to 1.0."
    assert numpy.array_equal(spectrum.peaks.intensities, intensities/100), "Expected different intensities"
    assert numpy.array_equal(spectrum.peaks.mz, mz), "Expected different peak mz."


def test_square_root_intensities_losses_present(spectrum_with_losses, mz, intensities):
    """Test if also losses (if present) are square root correctly."""

    spectrum = normalize_intensities(spectrum_with_losses)

    expected_loss_intensities = numpy.array([1., 0.1, 0.01, 0.], dtype='float')

    assert max(spectrum.peaks.intensities) == 1.0, "Expected the spectrum to be scaled to 1.0."
    assert numpy.array_equal(spectrum.peaks.intensities, intensities/100), "Expected different intensities"
    assert max(spectrum.losses.intensities) == 1.0, "Expected the losses to be scaled to 1.0."
    assert numpy.array_equal(spectrum.losses.intensities, expected_loss_intensities), "Expected different loss intensities"
    assert numpy.array_equal(spectrum.peaks.mz, mz), "Expected different peak mz."


def test_normalize_intensities_empty_peaks(spectrum_without_peaks):
    """Test running filter with empty peaks spectrum."""

    spectrum = normalize_intensities(spectrum_without_peaks)

    assert spectrum == spectrum_without_peaks, "Spectrum should remain unchanged."


def test_normalize_intensities_empty_spectrum():
    """Test running filter with spectrum == None."""
    spectrum = normalize_intensities(None)

    assert spectrum is None, "Expected spectrum to be None."
