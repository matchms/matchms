import numpy as np
import pytest
from testfixtures import LogCapture
from matchms.filtering import reduce_to_number_of_peaks
from matchms.logging_functions import reset_matchms_logger, set_matchms_logger_level
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata", [{}, {"parent_mass": 50}])
def test_reduce_to_number_of_peaks_no_changes(metadata):
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = reduce_to_number_of_peaks(spectrum_in)

    assert spectrum == spectrum_in, "Expected no changes."


@pytest.mark.parametrize(
    "mz, intensities, metadata, params, expected",
    [
        [np.array([10, 20, 30, 40, 50], dtype="float"), np.array([1, 1, 10, 20, 100], dtype="float"), {}, [1, 4, None], [20.0, 30.0, 40.0, 50.0]],
        [np.array([10, 20, 30, 40], dtype="float"), np.array([0, 1, 10, 100], dtype="float"), {"parent_mass": 20}, [2, 4, 0.1], [30.0, 40.0]],
        [np.array([10, 20, 30, 40], dtype="float"), np.array([0, 1, 10, 100], dtype="float"), {"parent_mass": 20}, [3, 4, 0.1], [20.0, 30.0, 40.0]],
        [
            np.array([10, 20, 30, 40, 50, 60], dtype="float"),
            np.array([1, 1, 10, 100, 50, 20], dtype="float"),
            {"parent_mass": 60},
            [3, 4, 0.1],
            [30.0, 40.0, 50.0, 60.0],
        ],
    ],
)
def test_reduce_to_number_of_peaks(mz, intensities, metadata, params, expected):
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()
    n_required, n_max, ratio_desired = params

    spectrum = reduce_to_number_of_peaks(spectrum_in, n_required=n_required, n_max=n_max, ratio_desired=ratio_desired)

    assert len(spectrum.peaks) == len(expected), "Expected that only 4 peaks remain."
    assert spectrum.peaks.mz.tolist() == expected, "Expected different peaks to remain."


def test_reduce_to_number_of_peaks_set_to_none():
    """Test is spectrum is set to None if not enough peaks."""
    set_matchms_logger_level("INFO")
    mz = np.array([10, 20], dtype="float")
    intensities = np.array([0.5, 1], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata({"parent_mass": 50}).build()

    with LogCapture() as log:
        spectrum = reduce_to_number_of_peaks(spectrum_in, n_required=5)

    assert spectrum is None, "Expected spectrum to be set to None."
    log.check(("matchms", "INFO", "Spectrum with 2 (<5) peaks was set to None."))
    reset_matchms_logger()


def test_reduce_to_number_of_peaks_n_max_4():
    """Test setting n_max parameter."""
    mz = np.array([10, 20, 30, 40, 50], dtype="float")
    intensities = np.array([1, 1, 10, 20, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = reduce_to_number_of_peaks(spectrum_in, n_max=4)

    expected = np.array([20, 30, 40, 50], dtype="float")

    assert len(spectrum.peaks) == len(expected), "Expected that only 4 peaks remain."
    np.testing.assert_array_equal(spectrum.peaks.mz, expected, err_msg="Expected different peaks to remain.")


def test_reduce_to_number_of_peaks_ratio_given_but_no_parent_mass():
    """A ratio_desired given without parent_mass should raise an exception."""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()
    with pytest.raises(Exception) as msg:
        _ = reduce_to_number_of_peaks(spectrum_in, n_required=4, ratio_desired=0.1)

    expected_msg = "Cannot use ratio_desired for spectrum without parent_mass."
    assert expected_msg in str(msg.value), "Expected specific exception message."


def test_reduce_to_number_of_peaks_desired_5_check_sorting():
    """Check if mz and intensities order is sorted correctly"""
    mz = np.array([10, 20, 30, 40, 50, 60], dtype="float")
    intensities = np.array([5, 1, 4, 3, 100, 2], dtype="float")
    metadata = {"parent_mass": 20}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = reduce_to_number_of_peaks(spectrum_in, n_max=5)

    assert spectrum.peaks.intensities.tolist() == [5.0, 4.0, 3.0, 100.0, 2.0], "Expected different intensities."
    assert spectrum.peaks.mz.tolist() == [10.0, 30.0, 40.0, 50.0, 60.0], "Expected different peaks to remain."


def test_empty_spectrum():
    spectrum_in = None
    spectrum = reduce_to_number_of_peaks(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
