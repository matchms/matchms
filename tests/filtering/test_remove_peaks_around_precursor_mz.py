import numpy as np
import pytest
from matchms.filtering import remove_peaks_around_precursor_mz
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.fixture
def spectrum_in():
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    metadata = {"precursor_mz": 60.0}
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )
    return spectrum_in


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_remove_peaks_around_precursor_mz_no_params(spectrum_in, as_collection):
    """Using defaults with precursor mz present."""
    spectrum = run_filter_as_spectrum_or_collection(
        remove_peaks_around_precursor_mz,
        spectrum_in,
        as_collection,
    )

    assert len(spectrum.peaks) == len(spectrum_in.peaks)
    np.testing.assert_allclose(spectrum.peaks.mz, spectrum_in.peaks.mz, atol=1e-6)
    np.testing.assert_array_equal(spectrum.peaks.intensities, spectrum_in.peaks.intensities)


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_remove_peaks_around_precursor_mz_tolerance_20(spectrum_in, as_collection):
    """Set mz_tolerance to 20."""
    spectrum = run_filter_as_spectrum_or_collection(
        remove_peaks_around_precursor_mz,
        spectrum_in,
        as_collection,
        mz_tolerance=20,
    )

    assert len(spectrum.peaks) == 3, "Expected 3 peaks to remain."
    np.testing.assert_allclose(
        spectrum.peaks.mz,
        np.array([10.0, 20.0, 30.0]),
        atol=1e-6,
        err_msg="Expected different peaks to remain.",
    )
    np.testing.assert_array_equal(
        spectrum.peaks.intensities,
        np.array([0.0, 1.0, 10.0]),
        err_msg="Expected different intensities to remain.",
    )


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": 1.0}).build()

    spectrum = remove_peaks_around_precursor_mz(spectrum_in)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), "Expected input spectrum to remain unchanged."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_remove_peaks_around_precursor_without_precursor_mz(spectrum_in, as_collection):
    """Test if correct error is raised for missing precursor_mz."""
    spectrum_in.metadata = {}

    with pytest.raises(ValueError, match="Undefined 'precursor_mz'."):
        run_filter_as_spectrum_or_collection(
            remove_peaks_around_precursor_mz,
            spectrum_in,
            as_collection,
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_remove_peaks_around_precursor_with_wrong_precursor_mz(spectrum_in, as_collection):
    """Test if correct error is raised for precursor_mz as string."""
    spectrum_in.set("precursor_mz", "445.0")

    with pytest.raises(ValueError, match="Expected 'precursor_mz' to be a scalar number."):
        run_filter_as_spectrum_or_collection(
            remove_peaks_around_precursor_mz,
            spectrum_in,
            as_collection,
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_if_precursor_remains(as_collection):
    """Test if peaks around precursor mz are removed, but precursor peak remains."""
    mz = np.array([10, 20, 400, 410.5], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    metadata = {"precursor_mz": 410.5}
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        remove_peaks_around_precursor_mz,
        spectrum_in,
        as_collection,
    )

    assert len(spectrum.peaks) == 3, "Expected 3 peaks to remain."
    np.testing.assert_allclose(
        spectrum.peaks.mz,
        np.array([10.0, 20.0, 410.5]),
        atol=1e-6,
        err_msg="Expected different peaks to remain.",
    )
    np.testing.assert_array_equal(
        spectrum.peaks.intensities,
        np.array([0.0, 1.0, 100.0]),
        err_msg="Expected different intensities to remain.",
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_remove_peaks_around_precursor_mz_negative_tolerance(spectrum_in, as_collection):
    """Test if negative mz_tolerance raises an error."""
    with pytest.raises(ValueError, match="mz_tolerance must be a positive scalar."):
        run_filter_as_spectrum_or_collection(
            remove_peaks_around_precursor_mz,
            spectrum_in,
            as_collection,
            mz_tolerance=-1,
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_remove_peaks_around_precursor_mz_tolerance_zero_keeps_all_peaks(as_collection):
    """With tolerance 0, only non-exact peaks within 0 Da would be removed, so all peaks remain."""
    mz = np.array([100.0, 101.0, 102.0], dtype="float")
    intensities = np.array([1.0, 2.0, 3.0], dtype="float")
    metadata = {"precursor_mz": 101.0}
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        remove_peaks_around_precursor_mz,
        spectrum_in,
        as_collection,
        mz_tolerance=0,
    )

    assert len(spectrum.peaks) == 3
    np.testing.assert_allclose(spectrum.peaks.mz, mz, atol=1e-6)
    np.testing.assert_array_equal(spectrum.peaks.intensities, intensities)


def test_with_input_none():
    """Test if input spectrum is None."""
    assert remove_peaks_around_precursor_mz(None) is None
