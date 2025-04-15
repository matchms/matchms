import numpy as np
import pytest
from matchms.filtering import remove_noise_below_frequent_intensities
from matchms.Spectrum import Spectrum


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
        ],  # test noise level multiplier
        [
            np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype="float"),
            np.array([20, 20, 20, 20, 20, 500, 10, 5, 100], dtype="float"),
            10,
            5.0,
            9,
        ],  # test expected_number_of_peaks
    ],
)
def test_remove_noise_below_frequent_intensities(
    mz, intensities, min_count_of_frequent_intensities, noise_level_multiplier, expected_number_of_peaks
):
    spectrum_in = Spectrum(mz, intensities)
    spectrum_out = remove_noise_below_frequent_intensities(
        spectrum_in,
        min_count_of_frequent_intensities=min_count_of_frequent_intensities,
        noise_level_multiplier=noise_level_multiplier,
    )

    assert len(spectrum_out.peaks.mz) == expected_number_of_peaks
