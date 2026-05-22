import numpy as np
import pytest
from matchms.filtering import select_by_mz
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "peaks, mz_from, mz_to, expected",
    [
        [
            [np.array([10, 20, 30, 40], dtype="float"), np.array([1, 10, 100, 1000], dtype="float")],
            0,
            1000,
            [np.array([10, 20, 30, 40], dtype="float"), np.array([1, 10, 100, 1000], dtype="float")],
        ],
        [
            [np.array([998, 999, 1000, 1001, 1002], dtype="float"), np.array([1, 10, 100, 1000, 10000], dtype="float")],
            0,
            1000,
            [np.array([998, 999, 1000], dtype="float"), np.array([1, 10, 100], dtype="float")],
        ],
        [
            [np.array([10, 20, 30, 40], dtype="float"), np.array([1, 10, 100, 1000], dtype="float")],
            15,
            1000,
            [np.array([20, 30, 40], dtype="float"), np.array([10, 100, 1000], dtype="float")],
        ],
        [
            [np.array([10, 20, 30, 40], dtype="float"), np.array([1, 10, 100, 1000], dtype="float")],
            0,
            35,
            [np.array([10, 20, 30], dtype="float"), np.array([1, 10, 100], dtype="float")],
        ],
        [
            [np.array([10, 20, 30, 40], dtype="float"), np.array([1, 10, 100, 1000], dtype="float")],
            15,
            35,
            [np.array([20, 30], dtype="float"), np.array([10, 100], dtype="float")],
        ],
        [
            # Inclusive boundaries.
            [np.array([10, 20, 30], dtype="float"), np.array([1, 10, 100], dtype="float")],
            10,
            30,
            [np.array([10, 20, 30], dtype="float"), np.array([1, 10, 100], dtype="float")],
        ],
        [
            # No peaks remain.
            [np.array([10, 20, 30], dtype="float"), np.array([1, 10, 100], dtype="float")],
            100,
            200,
            [np.array([], dtype="float"), np.array([], dtype="float")],
        ],
    ],
)
def test_select_by_mz(peaks, mz_from, mz_to, expected, as_collection):
    spectrum_in = SpectrumBuilder().with_mz(peaks[0]).with_intensities(peaks[1]).build()

    spectrum = run_filter_as_spectrum_or_collection(
        select_by_mz,
        spectrum_in,
        as_collection,
        mz_from=mz_from,
        mz_to=mz_to,
    )

    assert spectrum.peaks.mz.size == len(expected[0])
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size

    # SpectraCollection reconstructs m/z values from binned storage, so exact
    # m/z equality can differ by half a bin. Intensities should remain exact.
    np.testing.assert_allclose(spectrum.peaks.mz, expected[0], atol=1e-6)
    np.testing.assert_array_equal(spectrum.peaks.intensities, expected[1])


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_select_by_mz_rejects_invalid_mz_range(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30], dtype="float"))
        .with_intensities(np.array([1, 10, 100], dtype="float"))
        .build()
    )

    with pytest.raises(
        ValueError,
        match="'mz_from' should be smaller than or equal to 'mz_to'",
    ):
        run_filter_as_spectrum_or_collection(
            select_by_mz,
            spectrum_in,
            as_collection,
            mz_from=100,
            mz_to=10,
        )


def test_select_by_mz_with_input_none():
    spectrum = select_by_mz(None)

    assert spectrum is None


def test_select_by_mz_clone_true_does_not_modify_input_spectrum():
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([1, 10, 100, 1000], dtype="float"))
        .build()
    )

    spectrum = select_by_mz(
        spectrum_in,
        mz_from=15,
        mz_to=35,
        clone=True,
    )

    assert spectrum is not spectrum_in
    np.testing.assert_array_equal(spectrum_in.peaks.mz, np.array([10, 20, 30, 40], dtype="float"))
    np.testing.assert_array_equal(spectrum_in.peaks.intensities, np.array([1, 10, 100, 1000], dtype="float"))
    np.testing.assert_array_equal(spectrum.peaks.mz, np.array([20, 30], dtype="float"))
    np.testing.assert_array_equal(spectrum.peaks.intensities, np.array([10, 100], dtype="float"))


def test_select_by_mz_clone_false_modifies_input_spectrum():
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([1, 10, 100, 1000], dtype="float"))
        .build()
    )

    spectrum = select_by_mz(
        spectrum_in,
        mz_from=15,
        mz_to=35,
        clone=False,
    )

    assert spectrum is spectrum_in
    np.testing.assert_array_equal(spectrum.peaks.mz, np.array([20, 30], dtype="float"))
    np.testing.assert_array_equal(spectrum.peaks.intensities, np.array([10, 100], dtype="float"))
