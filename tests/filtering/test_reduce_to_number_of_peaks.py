import numpy as np
import pytest
from testfixtures import LogCapture
from matchms import SpectraCollection
from matchms.filtering import reduce_to_number_of_peaks
from matchms.logging_functions import reset_matchms_logger, set_matchms_logger_level
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize("metadata", [{}, {"parent_mass": 50}])
def test_reduce_to_number_of_peaks_no_changes(metadata, as_collection):
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        reduce_to_number_of_peaks,
        spectrum_in,
        as_collection,
    )

    assert len(spectrum.peaks) == len(spectrum_in.peaks)
    np.testing.assert_allclose(spectrum.peaks.mz, spectrum_in.peaks.mz, atol=1e-6)
    np.testing.assert_array_equal(spectrum.peaks.intensities, spectrum_in.peaks.intensities)


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "mz, intensities, metadata, params, expected",
    [
        [
            np.array([10, 20, 30, 40, 50], dtype="float"),
            np.array([1, 1, 10, 20, 100], dtype="float"),
            {},
            [1, 4, None],
            [20.0, 30.0, 40.0, 50.0],
        ],
        [
            np.array([10, 20, 30, 40], dtype="float"),
            np.array([0, 1, 10, 100], dtype="float"),
            {"parent_mass": 20},
            [2, 4, 0.1],
            [30.0, 40.0],
        ],
        [
            np.array([10, 20, 30, 40], dtype="float"),
            np.array([0, 1, 10, 100], dtype="float"),
            {"parent_mass": 20},
            [3, 4, 0.1],
            [20.0, 30.0, 40.0],
        ],
        [
            np.array([10, 20, 30, 40, 50, 60], dtype="float"),
            np.array([1, 1, 10, 100, 50, 20], dtype="float"),
            {"parent_mass": 60},
            [3, 4, 0.1],
            [30.0, 40.0, 50.0, 60.0],
        ],
    ],
)
def test_reduce_to_number_of_peaks(mz, intensities, metadata, params, expected, as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )
    n_required, n_max, ratio_desired = params

    spectrum = run_filter_as_spectrum_or_collection(
        reduce_to_number_of_peaks,
        spectrum_in,
        as_collection,
        n_required=n_required,
        n_max=n_max,
        ratio_desired=ratio_desired,
    )

    assert len(spectrum.peaks) == len(expected), f"Expected that only {len(expected)} peaks remain."
    np.testing.assert_allclose(
        spectrum.peaks.mz,
        expected,
        atol=1e-6,
        err_msg="Expected different peaks to remain.",
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_reduce_to_number_of_peaks_set_to_none(as_collection):
    """Test if spectrum is set to None if not enough peaks."""
    set_matchms_logger_level("INFO")
    mz = np.array([10, 20], dtype="float")
    intensities = np.array([0.5, 1], dtype="float")
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata({"parent_mass": 50})
        .build()
    )

    with LogCapture() as log:
        spectrum = run_filter_as_spectrum_or_collection(
            reduce_to_number_of_peaks,
            spectrum_in,
            as_collection,
            n_required=5,
        )

    assert spectrum is None, "Expected spectrum to be set to None."

    if as_collection:
        log.check(("matchms", "INFO", "All spectra had fewer than 5 peaks and were removed."))
    else:
        log.check(("matchms", "INFO", "Spectrum with 2 (<5) peaks was set to None."))

    reset_matchms_logger()


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_reduce_to_number_of_peaks_n_max_4(as_collection):
    """Test setting n_max parameter."""
    mz = np.array([10, 20, 30, 40, 50], dtype="float")
    intensities = np.array([1, 1, 10, 20, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = run_filter_as_spectrum_or_collection(
        reduce_to_number_of_peaks,
        spectrum_in,
        as_collection,
        n_max=4,
    )

    expected = np.array([20, 30, 40, 50], dtype="float")

    assert len(spectrum.peaks) == len(expected), "Expected that only 4 peaks remain."
    np.testing.assert_allclose(
        spectrum.peaks.mz,
        expected,
        atol=1e-6,
        err_msg="Expected different peaks to remain.",
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_reduce_to_number_of_peaks_ratio_given_but_no_parent_mass(as_collection):
    """A ratio_desired given without parent_mass should raise an exception."""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    with pytest.raises(Exception) as msg:
        run_filter_as_spectrum_or_collection(
            reduce_to_number_of_peaks,
            spectrum_in,
            as_collection,
            n_required=4,
            ratio_desired=0.1,
        )

    expected_msg = "Cannot use ratio_desired for spectrum without parent_mass."
    assert expected_msg in str(msg.value), "Expected specific exception message."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_reduce_to_number_of_peaks_ratio_given_but_parent_mass_is_none(as_collection):
    """A ratio_desired given with parent_mass=None should raise an exception."""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata({"parent_mass": None})
        .build()
    )

    with pytest.raises(Exception) as msg:
        run_filter_as_spectrum_or_collection(
            reduce_to_number_of_peaks,
            spectrum_in,
            as_collection,
            n_required=4,
            ratio_desired=0.1,
        )

    expected_msg = "Cannot use ratio_desired for spectrum without parent_mass."
    assert expected_msg in str(msg.value), "Expected specific exception message."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_reduce_to_number_of_peaks_desired_5_check_sorting(as_collection):
    """Check if mz and intensities order is sorted correctly."""
    mz = np.array([10, 20, 30, 40, 50, 60], dtype="float")
    intensities = np.array([5, 1, 4, 3, 100, 2], dtype="float")
    metadata = {"parent_mass": 20}
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        reduce_to_number_of_peaks,
        spectrum_in,
        as_collection,
        n_max=5,
    )

    np.testing.assert_array_equal(
        spectrum.peaks.intensities,
        [5.0, 4.0, 3.0, 100.0, 2.0],
        err_msg="Expected different intensities.",
    )
    np.testing.assert_allclose(
        spectrum.peaks.mz,
        [10.0, 30.0, 40.0, 50.0, 60.0],
        atol=1e-6,
        err_msg="Expected different peaks to remain.",
    )


def test_reduce_to_number_of_peaks_collection_drops_only_rows_with_too_few_peaks():
    """Collection implementation should drop only spectra with fewer than n_required peaks."""
    spectrum_too_few_peaks = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20], dtype="float"))
        .with_intensities(np.array([1, 2], dtype="float"))
        .with_metadata({"id": "too_few"})
        .build()
    )
    spectrum_enough_peaks = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([1, 2, 3, 4], dtype="float"))
        .with_metadata({"id": "enough"})
        .build()
    )

    collection = SpectraCollection([spectrum_too_few_peaks, spectrum_enough_peaks])

    reduced = reduce_to_number_of_peaks(collection, n_required=3)

    assert isinstance(reduced, SpectraCollection)
    assert len(reduced) == 1
    assert reduced.metadata.loc[0, "id"] == "enough"
    np.testing.assert_allclose(reduced[0].peaks.mz, [10, 20, 30, 40], atol=1e-6)


def test_reduce_to_number_of_peaks_collection_with_row_specific_ratio_desired():
    """Collection implementation should compute ratio_desired per row from parent_mass."""
    spectrum_1 = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([1, 2, 3, 4], dtype="float"))
        .with_metadata({"id": "s1", "parent_mass": 20})
        .build()
    )
    spectrum_2 = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30, 40, 50, 60], dtype="float"))
        .with_intensities(np.array([1, 2, 3, 4, 5, 6], dtype="float"))
        .with_metadata({"id": "s2", "parent_mass": 50})
        .build()
    )

    collection = SpectraCollection([spectrum_1, spectrum_2])

    reduced = reduce_to_number_of_peaks(
        collection,
        n_required=1,
        n_max=10,
        ratio_desired=0.1,
    )

    assert isinstance(reduced, SpectraCollection)
    assert len(reduced) == 2

    # parent_mass 20, ratio 0.1 -> ceil(2.0) -> keep 2 highest peaks
    np.testing.assert_allclose(reduced[0].peaks.mz, [30, 40], atol=1e-6)
    np.testing.assert_array_equal(reduced[0].peaks.intensities, [3, 4])

    # parent_mass 50, ratio 0.1 -> ceil(5.0) -> keep 5 highest peaks
    np.testing.assert_allclose(reduced[1].peaks.mz, [20, 30, 40, 50, 60], atol=1e-6)
    np.testing.assert_array_equal(reduced[1].peaks.intensities, [2, 3, 4, 5, 6])


def test_reduce_to_number_of_peaks_collection_ratio_desired_missing_parent_mass_in_one_row():
    """A missing parent_mass in any row should raise when ratio_desired is used."""
    spectrum_with_parent_mass = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([1, 2, 3, 4], dtype="float"))
        .with_metadata({"parent_mass": 20})
        .build()
    )
    spectrum_without_parent_mass = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([1, 2, 3, 4], dtype="float"))
        .with_metadata({})
        .build()
    )

    collection = SpectraCollection([spectrum_with_parent_mass, spectrum_without_parent_mass])

    with pytest.raises(ValueError, match="Cannot use ratio_desired for spectrum without parent_mass."):
        reduce_to_number_of_peaks(collection, n_required=1, ratio_desired=0.1)


def test_empty_spectrum():
    spectrum_in = None
    spectrum = reduce_to_number_of_peaks(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
