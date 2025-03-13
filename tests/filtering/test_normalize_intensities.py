import numpy as np
import pytest
from matchms.filtering import normalize_intensities
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("mz, intensities", [
    [np.array([10, 20, 30, 40], dtype='float'), np.array([0, 1, 10, 100], dtype='float')]
])
def test_normalize_intensities(mz, intensities):
    """Test if peak intensities are normalized correctly."""
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = normalize_intensities(spectrum_in)

    assert max(spectrum.peaks.intensities) == 1.0, "Expected the spectrum to be scaled to 1.0."
    assert np.array_equal(spectrum.peaks.intensities, intensities/100), "Expected different intensities"
    assert np.array_equal(spectrum.peaks.mz, mz), "Expected different peak mz."


@pytest.mark.parametrize("mz, intensities, metadata, expected_losses", [
    [np.array([10, 20, 30, 40], dtype='float'), np.array([0, 1, 10, 100], dtype='float'),
     {"precursor_mz": 45.0}, np.array([1., 0.1, 0.01, 0.], dtype='float')]
])
def test_normalize_intensities_losses_present(mz, intensities, metadata, expected_losses):
    """Test if also losses (if present) are normalized correctly."""
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(
        intensities).with_metadata(metadata).build()

    spectrum = normalize_intensities(spectrum_in)

    assert max(spectrum.peaks.intensities) == 1.0, "Expected the spectrum to be scaled to 1.0."
    assert np.array_equal(spectrum.peaks.intensities, intensities/100), "Expected different intensities"
    assert max(spectrum.losses.intensities) == 1.0, "Expected the losses to be scaled to 1.0."
    assert np.all(spectrum.losses.intensities == expected_losses), "Expected different loss intensities"


def test_normalize_intensities_empty_peaks():
    """Test running filter with empty peaks spectrum."""
    spectrum_in = SpectrumBuilder().build()

    spectrum = normalize_intensities(spectrum_in)

    assert spectrum == spectrum_in, "Spectrum should remain unchanged."


def test_normalize_intensities_all_zeros(caplog):
    """Test if non-sense intensities are handled correctly."""
    mz = np.array([10, 20, 30], dtype='float')
    intensities = np.array([0, 0, 0], dtype='float')
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = normalize_intensities(spectrum_in)

    assert len(spectrum.peaks.intensities) == 0, "Expected no peak intensities."
    assert len(spectrum.peaks.mz) == 0, "Expected no m/z values. "
    msg = "Peaks of spectrum with all peak intensities <= 0 were deleted."
    assert msg in caplog.text, "Expected log message."


def test_normalize_intensities_empty_spectrum():
    """Test running filter with spectrum == None."""
    spectrum = normalize_intensities(None)

    assert spectrum is None, "Expected spectrum to be None."
