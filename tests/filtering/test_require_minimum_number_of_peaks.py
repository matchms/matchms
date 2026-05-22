import numpy as np
import pytest
from matchms import SpectraCollection
from matchms.filtering import require_minimum_number_of_peaks
from matchms.typing import SpectrumType
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.fixture
def spectrum_in():
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    metadata = {"parent_mass": 10}
    return (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_require_minimum_number_of_peaks_no_params(spectrum_in: SpectrumType, as_collection):
    spectrum = run_filter_as_spectrum_or_collection(
        require_minimum_number_of_peaks,
        spectrum_in,
        as_collection,
    )

    assert spectrum is None


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_require_minimum_number_of_peaks_required_4(spectrum_in: SpectrumType, as_collection):
    spectrum = run_filter_as_spectrum_or_collection(
        require_minimum_number_of_peaks,
        spectrum_in,
        as_collection,
        n_required=4,
    )

    assert spectrum is not None
    assert len(spectrum.peaks) == 4


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_require_minimum_number_of_peaks_required_4_or_1_no_parent_mass(
    spectrum_in: SpectrumType,
    as_collection,
):
    spectrum_in.set("parent_mass", None)

    spectrum = run_filter_as_spectrum_or_collection(
        require_minimum_number_of_peaks,
        spectrum_in,
        as_collection,
        n_required=4,
        ratio_required=0.1,
    )

    assert spectrum is not None
    assert len(spectrum.peaks) == 4


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_require_minimum_number_of_peaks_required_4_or_1(
    spectrum_in: SpectrumType,
    as_collection,
):
    spectrum = run_filter_as_spectrum_or_collection(
        require_minimum_number_of_peaks,
        spectrum_in,
        as_collection,
        n_required=4,
        ratio_required=0.1,
    )

    assert spectrum is not None
    assert len(spectrum.peaks) == 4


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_require_minimum_number_of_peaks_required_4_ratio_none(
    spectrum_in: SpectrumType,
    as_collection,
):
    """Test if parent_mass scaling is ignored when not passing ratio_required."""
    spectrum_in.set("parent_mass", 100)

    spectrum = run_filter_as_spectrum_or_collection(
        require_minimum_number_of_peaks,
        spectrum_in,
        as_collection,
        n_required=4,
    )

    assert spectrum is not None
    assert len(spectrum.peaks) == 4


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_require_minimum_number_of_peaks_required_4_or_10(
    spectrum_in: SpectrumType,
    as_collection,
):
    spectrum_in.set("parent_mass", 100)

    spectrum = run_filter_as_spectrum_or_collection(
        require_minimum_number_of_peaks,
        spectrum_in,
        as_collection,
        n_required=4,
        ratio_required=0.1,
    )

    assert spectrum is None


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_require_minimum_number_of_peaks_required_5_or_1(
    spectrum_in: SpectrumType,
    as_collection,
):
    spectrum = run_filter_as_spectrum_or_collection(
        require_minimum_number_of_peaks,
        spectrum_in,
        as_collection,
        n_required=5,
        ratio_required=0.1,
    )

    assert spectrum is None


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_require_minimum_number_of_peaks_required_5_or_10(
    spectrum_in: SpectrumType,
    as_collection,
):
    spectrum_in.set("parent_mass", 100)

    spectrum = run_filter_as_spectrum_or_collection(
        require_minimum_number_of_peaks,
        spectrum_in,
        as_collection,
        n_required=5,
        ratio_required=0.1,
    )

    assert spectrum is None


def test_require_minimum_number_of_peaks_collection_drops_only_failing_rows():
    spectra = [
        SpectrumBuilder()
        .with_metadata({"id": "keep", "parent_mass": 10})
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([0, 1, 10, 100], dtype="float"))
        .build(),
        SpectrumBuilder()
        .with_metadata({"id": "drop", "parent_mass": 10})
        .with_mz(np.array([10, 20], dtype="float"))
        .with_intensities(np.array([0, 1], dtype="float"))
        .build(),
    ]
    collection = SpectraCollection(spectra)

    filtered = require_minimum_number_of_peaks(collection, n_required=4)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == 1
    assert filtered.metadata.loc[0, "id"] == "keep"


def test_require_minimum_number_of_peaks_collection_returns_none_if_all_rows_fail():
    spectra = [
        SpectrumBuilder()
        .with_metadata({"id": "drop1"})
        .with_mz(np.array([10, 20], dtype="float"))
        .with_intensities(np.array([0, 1], dtype="float"))
        .build(),
        SpectrumBuilder()
        .with_metadata({"id": "drop2"})
        .with_mz(np.array([10], dtype="float"))
        .with_intensities(np.array([0], dtype="float"))
        .build(),
    ]
    collection = SpectraCollection(spectra)

    filtered = require_minimum_number_of_peaks(collection, n_required=4)

    assert filtered is None


def test_empty_spectrum():
    assert require_minimum_number_of_peaks(None) is None
