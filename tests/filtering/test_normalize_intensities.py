import numpy as np
import pytest
from matchms.filtering import normalize_intensities
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "mz, intensities, expected_mz, expected_intensities",
    [
        [
            np.array([10, 20, 30, 40], dtype="float"),
            np.array([0, 1, 10, 100], dtype="float"),
            np.array([20, 30, 40], dtype="float"),
            np.array([0.01, 0.1, 1.0], dtype="float"),
        ]
    ],
)
def test_normalize_intensities(mz, intensities, expected_mz, expected_intensities, as_collection):
    """Test if peak intensities are normalized correctly."""
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = run_filter_as_spectrum_or_collection(
        normalize_intensities,
        spectrum_in,
        as_collection,
    )

    assert max(spectrum.peaks.intensities) == 1.0, "Expected the spectrum to be scaled to 1.0."
    np.testing.assert_allclose(
        spectrum.peaks.intensities,
        expected_intensities,
        err_msg="Expected different intensities",
    )
    np.testing.assert_allclose(
        spectrum.peaks.mz,
        expected_mz,
        atol=1e-6,
        err_msg="Expected different peak mz.",
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "mz, intensities, metadata, expected_losses",
    [
        [
            np.array([10, 20, 30, 40], dtype="float"),
            np.array([0, 1, 10, 100], dtype="float"),
            {"precursor_mz": 45.0},
            np.array([1.0, 0.1, 0.01], dtype="float"),
        ]
    ],
)
def test_normalize_intensities_losses_present(mz, intensities, metadata, expected_losses, as_collection):
    """Test if also losses (if present) are normalized correctly."""
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        normalize_intensities,
        spectrum_in,
        as_collection,
    )

    assert max(spectrum.peaks.intensities) == 1.0, "Expected the spectrum to be scaled to 1.0."
    np.testing.assert_allclose(
        spectrum.peaks.intensities,
        np.array([0.01, 0.1, 1.0], dtype="float"),
        err_msg="Expected different intensities",
    )

    assert max(spectrum.losses.intensities) == 1.0, "Expected the losses to be scaled to 1.0."
    np.testing.assert_allclose(
        spectrum.losses.intensities,
        expected_losses,
        err_msg="Expected different loss intensities",
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_normalize_intensities_empty_peaks(as_collection):
    """Test running filter with empty peaks spectrum."""
    spectrum_in = SpectrumBuilder().build()

    spectrum = run_filter_as_spectrum_or_collection(
        normalize_intensities,
        spectrum_in,
        as_collection,
    )

    assert len(spectrum.peaks) == 0, "Spectrum should remain without peaks."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_normalize_intensities_all_zeros(caplog, as_collection):
    """Test if non-sense intensities are handled correctly."""
    mz = np.array([10, 20, 30], dtype="float")
    intensities = np.array([0, 0, 0], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = run_filter_as_spectrum_or_collection(
        normalize_intensities,
        spectrum_in,
        as_collection,
    )

    assert len(spectrum.peaks.intensities) == 0, "Expected no peak intensities."
    assert len(spectrum.peaks.mz) == 0, "Expected no m/z values."
    msg = "Peaks of spectrum with all peak intensities <= 0 were deleted."
    assert msg in caplog.text, "Expected log message."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_normalize_intensities_rejects_negative_intensities(as_collection):
    """Test if negative intensities are rejected."""
    mz = np.array([10, 20, 30], dtype="float")
    intensities = np.array([1, -1, 10], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    with pytest.raises(ValueError, match="Negative peak intensities are not allowed"):
        run_filter_as_spectrum_or_collection(
            normalize_intensities,
            spectrum_in,
            as_collection,
        )


def test_normalize_intensities_empty_spectrum():
    """Test running filter with spectrum == None."""
    spectrum = normalize_intensities(None)

    assert spectrum is None, "Expected spectrum to be None."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "mz, intensities, scale_to_max, expected_intensities",
    [
        # Test case 1: Standard normalization
        [
            np.array([10.0, 20.0, 30.0, 40.0], dtype="float"),
            np.array([100.0, 200.0, 300.0, 400.0], dtype="float"),
            1.0,
            np.array([0.25, 0.5, 0.75, 1.0]),
        ],
        # Test case 2: Scale base peak to 100
        [
            np.array([10.0, 20.0, 30.0, 40.0], dtype="float"),
            np.array([100.0, 200.0, 300.0, 400.0], dtype="float"),
            100.0,
            np.array([25.0, 50.0, 75.0, 100.0]),
        ],
        # Test case 3: Scale base peak to 1000
        [
            np.array([50.0, 60.0, 70.0], dtype="float"),
            np.array([10.0, 50.0, 100.0], dtype="float"),
            1000.0,
            np.array([100.0, 500.0, 1000.0]),
        ],
        # Test case 4: Custom maximum
        [
            np.array([100.0, 200.0], dtype="float"),
            np.array([25.0, 75.0], dtype="float"),
            90.0,
            np.array([30.0, 90.0]),
        ],
        # Test case 5: Single peak
        [
            np.array([150.0], dtype="float"),
            np.array([500.0], dtype="float"),
            100.0,
            np.array([100.0]),
        ],
    ],
)
def test_normalize_intensities_scale_to_max(mz, intensities, scale_to_max, expected_intensities, as_collection):
    """Test normalize_intensities with various scale_to_max values."""
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = run_filter_as_spectrum_or_collection(
        normalize_intensities,
        spectrum_in,
        as_collection,
        scale_to_max=scale_to_max,
    )
    result_intensities = spectrum.peaks.intensities

    np.testing.assert_allclose(result_intensities, expected_intensities, atol=1e-2)

    original_order = np.argsort(intensities)
    result_order = np.argsort(result_intensities)
    np.testing.assert_array_equal(original_order, result_order)


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize("scale_to_max", [0, -1, -100.0])
def test_normalize_intensities_rejects_non_positive_scale_to_max(scale_to_max, as_collection):
    """Test if invalid scale_to_max values are rejected."""
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(np.array([10.0, 20.0]))
        .with_intensities(np.array([1.0, 10.0]))
        .build()
    )

    with pytest.raises(ValueError, match="'scale_to_max' must be > 0"):
        run_filter_as_spectrum_or_collection(
            normalize_intensities,
            spectrum_in,
            as_collection,
            scale_to_max=scale_to_max,
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize("scale_to_max", ["100", None, (0, 100)])
def test_normalize_intensities_rejects_non_numeric_scale_to_max(scale_to_max, as_collection):
    """Test if non-numeric scale_to_max values are rejected."""
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(np.array([10.0, 20.0]))
        .with_intensities(np.array([1.0, 10.0]))
        .build()
    )

    with pytest.raises(TypeError, match="'scale_to_max' must be a positive number"):
        run_filter_as_spectrum_or_collection(
            normalize_intensities,
            spectrum_in,
            as_collection,
            scale_to_max=scale_to_max,
        )