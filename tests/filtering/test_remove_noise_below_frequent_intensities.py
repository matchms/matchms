import numpy as np
import pytest
from matchms.filtering import remove_noise_below_frequent_intensities
from matchms.Spectrum import Spectrum
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "mz, intensities, min_count_of_frequent_intensities, noise_level_multiplier, expected_number_of_peaks",
    [
        [
            np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype="float"),
            np.array([20, 20, 20, 20, 20, 500, 10, 5, 20.1], dtype="float"),
            5,
            2.0,
            1,
        ],
        [
            np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype="float"),
            np.array([20, 20, 20, 20, 20, 500, 10, 5, 100], dtype="float"),
            5,
            5.0,
            1,
        ],
        [
            np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype="float"),
            np.array([20, 20, 20, 20, 20, 500, 10, 5, 100], dtype="float"),
            10,
            5.0,
            9,
        ],
    ],
)
def test_remove_noise_below_frequent_intensities(
    mz,
    intensities,
    min_count_of_frequent_intensities,
    noise_level_multiplier,
    expected_number_of_peaks,
    as_collection,
):
    spectrum_in = Spectrum(mz, intensities)

    spectrum_out = run_filter_as_spectrum_or_collection(
        remove_noise_below_frequent_intensities,
        spectrum_in,
        as_collection,
        min_count_of_frequent_intensities=min_count_of_frequent_intensities,
        noise_level_multiplier=noise_level_multiplier,
    )

    assert len(spectrum_out.peaks.mz) == expected_number_of_peaks
    assert len(spectrum_out.peaks.mz) == len(spectrum_out.peaks.intensities)


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_remove_noise_below_frequent_intensities_keeps_mz_and_intensities_aligned(as_collection):
    spectrum_in = Spectrum(
        mz=np.array([10, 20, 30, 40, 50, 60], dtype="float"),
        intensities=np.array([5, 5, 5, 5, 100, 200], dtype="float"),
    )

    spectrum_out = run_filter_as_spectrum_or_collection(
        remove_noise_below_frequent_intensities,
        spectrum_in,
        as_collection,
        min_count_of_frequent_intensities=4,
        noise_level_multiplier=2.0,
    )

    assert len(spectrum_out.peaks.mz) == 2
    np.testing.assert_allclose(
        spectrum_out.peaks.mz,
        np.array([50, 60], dtype="float"),
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        spectrum_out.peaks.intensities,
        np.array([100, 200], dtype="float"),
    )


def test_remove_noise_below_frequent_intensities_with_input_none():
    spectrum_out = remove_noise_below_frequent_intensities(None)

    assert spectrum_out is None
