import numpy as np
import pytest
from matchms import Spectrum
from matchms.filtering import select_by_relative_intensity
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.fixture
def spectrum_in() -> Spectrum:
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([1, 10, 100, 1000], dtype="float")
    return SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "intensity_from, intensity_to, expected_mz, expected_intensities",
    [
        [0, 1, np.array([10, 20, 30, 40], dtype="float"), np.array([1, 10, 100, 1000], dtype="float")],
        [0.01, 1, np.array([20, 30, 40], dtype="float"), np.array([10, 100, 1000], dtype="float")],
        [0, 0.99, np.array([10, 20, 30], dtype="float"), np.array([1, 10, 100], dtype="float")],
        [0.01, 0.99, np.array([20, 30], dtype="float"), np.array([10, 100], dtype="float")],
        # Inclusive boundaries: relative intensities are [0.001, 0.01, 0.1, 1.0].
        [0.01, 0.1, np.array([20, 30], dtype="float"), np.array([10, 100], dtype="float")],
        # No peaks remain.
        [0.2, 0.9, np.array([], dtype="float"), np.array([], dtype="float")],
    ],
)
def test_select_by_relative_intensity(
    spectrum_in,
    intensity_from,
    intensity_to,
    expected_mz,
    expected_intensities,
    as_collection,
):
    spectrum = run_filter_as_spectrum_or_collection(
        select_by_relative_intensity,
        spectrum_in,
        as_collection,
        intensity_from=intensity_from,
        intensity_to=intensity_to,
    )

    assert spectrum.peaks.mz.size == len(expected_mz)
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size

    # SpectraCollection reconstructs m/z values from binned storage, so exact m/z
    # equality can differ by half a bin. Intensities should remain exact.
    np.testing.assert_allclose(spectrum.peaks.mz, expected_mz, atol=1e-6)
    np.testing.assert_array_equal(spectrum.peaks.intensities, expected_intensities)


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_select_by_relative_intensity_with_from_parameter_too_small(
    spectrum_in: Spectrum,
    as_collection,
):
    with pytest.raises(
        ValueError,
        match="'intensity_from' should be larger than or equal to 0",
    ):
        run_filter_as_spectrum_or_collection(
            select_by_relative_intensity,
            spectrum_in,
            as_collection,
            intensity_from=-10.0,
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_select_by_relative_intensity_with_to_parameter_too_large(
    spectrum_in: Spectrum,
    as_collection,
):
    with pytest.raises(
        ValueError,
        match="'intensity_to' should be smaller than or equal to 1.0",
    ):
        run_filter_as_spectrum_or_collection(
            select_by_relative_intensity,
            spectrum_in,
            as_collection,
            intensity_to=10.0,
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_select_by_relative_intensity_with_from_larger_than_to(
    spectrum_in: Spectrum,
    as_collection,
):
    with pytest.raises(
        ValueError,
        match="'intensity_from' should be smaller than or equal to 'intensity_to'",
    ):
        run_filter_as_spectrum_or_collection(
            select_by_relative_intensity,
            spectrum_in,
            as_collection,
            intensity_from=0.8,
            intensity_to=0.2,
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_select_by_relative_intensity_with_empty_peaks(as_collection):
    """Spectra with empty peak arrays should not break."""
    spectrum_in = SpectrumBuilder().build()

    spectrum = run_filter_as_spectrum_or_collection(
        select_by_relative_intensity,
        spectrum_in,
        as_collection,
        intensity_from=0.01,
        intensity_to=0.99,
    )

    assert len(spectrum.peaks) == 0


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_select_by_relative_intensity_with_zero_intensity_peaks(as_collection):
    """Relative intensity is undefined if all intensities are zero.

    The refactored implementation removes such peaks instead of dividing by zero.
    """
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30], dtype="float"))
        .with_intensities(np.array([0, 0, 0], dtype="float"))
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        select_by_relative_intensity,
        spectrum_in,
        as_collection,
        intensity_from=0.0,
        intensity_to=1.0,
    )

    assert len(spectrum.peaks) == 0


def test_select_by_relative_intensity_with_input_none():
    spectrum = select_by_relative_intensity(None)

    assert spectrum is None


def test_select_by_relative_intensity_clone_true_does_not_modify_input_spectrum():
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([1, 10, 100, 1000], dtype="float"))
        .build()
    )

    spectrum = select_by_relative_intensity(
        spectrum_in,
        intensity_from=0.01,
        intensity_to=0.99,
        clone=True,
    )

    assert spectrum is not spectrum_in
    np.testing.assert_array_equal(spectrum_in.peaks.mz, np.array([10, 20, 30, 40], dtype="float"))
    np.testing.assert_array_equal(spectrum_in.peaks.intensities, np.array([1, 10, 100, 1000], dtype="float"))
    np.testing.assert_array_equal(spectrum.peaks.mz, np.array([20, 30], dtype="float"))
    np.testing.assert_array_equal(spectrum.peaks.intensities, np.array([10, 100], dtype="float"))


def test_select_by_relative_intensity_clone_false_modifies_input_spectrum():
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([1, 10, 100, 1000], dtype="float"))
        .build()
    )

    spectrum = select_by_relative_intensity(
        spectrum_in,
        intensity_from=0.01,
        intensity_to=0.99,
        clone=False,
    )

    assert spectrum is spectrum_in
    np.testing.assert_array_equal(spectrum.peaks.mz, np.array([20, 30], dtype="float"))
    np.testing.assert_array_equal(spectrum.peaks.intensities, np.array([10, 100], dtype="float"))
