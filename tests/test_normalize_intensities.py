import numpy
from matchms import Spectrum
from matchms.filtering import add_losses
from matchms.filtering import normalize_intensities


def test_normalize_intensities():
    """Test if peak intensities are normalized correctly."""
    mz = numpy.array([10, 20, 30, 40], dtype='float')
    intensities = numpy.array([0, 1, 10, 100], dtype='float')
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = normalize_intensities(spectrum_in)

    assert max(spectrum.peaks.intensities) == 1.0, "Expected the spectrum to be scaled to 1.0."
    assert numpy.array_equal(spectrum.peaks.intensities, intensities/100), "Expected different intensities"
    assert numpy.array_equal(spectrum.peaks.mz, mz), "Expected different peak mz."


def test_normalize_intensities_losses_present():
    """Test if also losses (if present) are normalized correctly."""
    mz = numpy.array([10, 20, 30, 40], dtype='float')
    intensities = numpy.array([0, 1, 10, 100], dtype='float')
    spectrum_in = Spectrum(mz=mz, intensities=intensities,
                           metadata={"precursor_mz": 45.0})

    spectrum = add_losses(spectrum_in)
    spectrum = normalize_intensities(spectrum)
    expected_loss_intensities = numpy.array([1., 0.1, 0.01, 0.], dtype='float')

    assert max(spectrum.peaks.intensities) == 1.0, "Expected the spectrum to be scaled to 1.0."
    assert numpy.array_equal(spectrum.peaks.intensities, intensities/100), "Expected different intensities"
    assert max(spectrum.losses.intensities) == 1.0, "Expected the losses to be scaled to 1.0."
    assert numpy.all(spectrum.losses.intensities == expected_loss_intensities), "Expected different loss intensities"


def test_normalize_intensities_empty_peaks():
    """Test running filter with empty peaks spectrum."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = normalize_intensities(spectrum_in)

    assert spectrum == spectrum_in, "Spectrum should remain unchanged."


def test_normalize_intensities_empty_spectrum():
    """Test running filter with spectrum == None."""
    spectrum = normalize_intensities(None)

    assert spectrum is None, "Expected spectrum to be None."
