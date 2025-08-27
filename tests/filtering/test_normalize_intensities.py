import numpy as np
import pytest
from matchms.filtering import normalize_intensities
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "mz, intensities", [[np.array([10, 20, 30, 40], dtype="float"), np.array([0, 1, 10, 100], dtype="float")]]
)
def test_normalize_intensities(mz, intensities):
    """Test if peak intensities are normalized correctly."""
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = normalize_intensities(spectrum_in)

    assert max(spectrum.peaks.intensities) == 1.0, "Expected the spectrum to be scaled to 1.0."
    assert np.array_equal(spectrum.peaks.intensities, intensities / 100), "Expected different intensities"
    assert np.array_equal(spectrum.peaks.mz, mz), "Expected different peak mz."


@pytest.mark.parametrize(
    "mz, intensities, metadata, expected_losses",
    [
        [
            np.array([10, 20, 30, 40], dtype="float"),
            np.array([0, 1, 10, 100], dtype="float"),
            {"precursor_mz": 45.0},
            np.array([1.0, 0.1, 0.01, 0.0], dtype="float"),
        ]
    ],
)
def test_normalize_intensities_losses_present(mz, intensities, metadata, expected_losses):
    """Test if also losses (if present) are normalized correctly."""
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = normalize_intensities(spectrum_in)

    assert max(spectrum.peaks.intensities) == 1.0, "Expected the spectrum to be scaled to 1.0."
    assert np.array_equal(spectrum.peaks.intensities, intensities / 100), "Expected different intensities"
    assert max(spectrum.losses.intensities) == 1.0, "Expected the losses to be scaled to 1.0."
    assert np.all(spectrum.losses.intensities == expected_losses), "Expected different loss intensities"


def test_normalize_intensities_empty_peaks():
    """Test running filter with empty peaks spectrum."""
    spectrum_in = SpectrumBuilder().build()

    spectrum = normalize_intensities(spectrum_in)

    assert spectrum == spectrum_in, "Spectrum should remain unchanged."


def test_normalize_intensities_all_zeros(caplog):
    """Test if non-sense intensities are handled correctly."""
    mz = np.array([10, 20, 30], dtype="float")
    intensities = np.array([0, 0, 0], dtype="float")
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


@pytest.mark.parametrize(
    "mz, intensities, scaling, expected_intensities",
    [
        # Test case 1: Standard normalization (no scaling)
        [
            np.array([10.0, 20.0, 30.0, 40.0], dtype="float"),
            np.array([100.0, 200.0, 300.0, 400.0], dtype="float"),
            None,
            np.array([0.25, 0.5, 0.75, 1.0]),
        ],
        # Test case 2: Scaling to 0-100
        [
            np.array([10.0, 20.0, 30.0, 40.0], dtype="float"),
            np.array([100.0, 200.0, 300.0, 400.0], dtype="float"),
            (0, 100),
            np.array([0, 33.33, 66.66, 100])
        ],
        # Test case 3: Scaling to 0-1000
        [
            np.array([50.0, 60.0, 70.0], dtype="float"),
            np.array([10.0, 50.0, 100.0], dtype="float"),
            (0, 1000),
            np.array([0, 444.44, 1000]),
        ],
        # Test case 4: Custom range scaling
        [
            np.array([100.0, 200.0], dtype="float"),
            np.array([25.0, 75.0], dtype="float"),
            (10, 90),
            np.array([10, 90])
        ],
        # Test case 5: Single peak
        [
            np.array([150.0], dtype="float"),
            np.array([500.0], dtype="float"),
            (0, 100),
            np.array([100.0]),  # Single value should be at max
        ],
    ],
)
def test_normalize_intensities_scaling(mz, intensities, scaling, expected_intensities):
    """Test normalize_intensities with various scaling parameters."""
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = normalize_intensities(spectrum_in, scaling=scaling)
    result_intensities = spectrum.peaks.intensities

    assert np.allclose(result_intensities, expected_intensities, atol=1e-2)

    original_order = np.argsort(intensities)
    result_order = np.argsort(result_intensities)
    np.testing.assert_array_equal(original_order, result_order)
