import numpy as np
import pytest
from matchms import SpectraCollection
from matchms.filtering.metadata_processing.require_precursor_mz import require_precursor_mz
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"precursor_mz": 60.0}, SpectrumBuilder().with_metadata({"precursor_mz": 60}).build()],
        [{"precursor_mz": 0.0}, None],
        [{"precursor_mz": -3.5}, None],
        [{}, None],
    ],
)
def test_require_precursor_mz_spectrum(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum == expected, "Expected no changes."


@pytest.mark.parametrize(
    "metadata, expected_kept",
    [
        [{"precursor_mz": 60.0}, True],
        [{"precursor_mz": 0.0}, False],
        [{"precursor_mz": -3.5}, False],
        [{}, False],
    ],
)
def test_require_precursor_mz_collection_single_row(metadata, expected_kept):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    collection = SpectraCollection([spectrum_in])

    filtered = require_precursor_mz(collection)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == int(expected_kept)


def test_require_precursor_mz_returns_same_spectrum_when_kept():
    """Requirement filters do not clone spectra; they only keep or remove them."""
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": 100.0}).build()

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum is spectrum_in


def test_require_precursor_mz_with_input_none():
    """Test if input spectrum is None."""
    spectrum = require_precursor_mz(None)

    assert spectrum is None


@pytest.mark.parametrize("precursor_mz", [0, 9.0, -200])
def test_require_precursor_mz_fail_when_mz_too_small(precursor_mz):
    """Test if spectrum is None when precursor_mz <= minimum_accepted_mz."""
    spectrum_in = SpectrumBuilder().build()
    spectrum_in.set("precursor_mz", precursor_mz)

    spectrum = require_precursor_mz(spectrum_in, minimum_accepted_mz=10)

    assert spectrum is None, "Expected spectrum to be None."


def test_require_precursor_mz_fail_because_below_zero():
    """Test if spectrum is None when precursor_mz < 0."""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()
    spectrum_in.set("precursor_mz", -3.5)

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum is None, "Expected spectrum to be None."


def test_require_precursor_mz_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"precursor_mz": 60.0}).build(),
            SpectrumBuilder().with_metadata({"precursor_mz": 0.0}).build(),
            SpectrumBuilder().with_metadata({"precursor_mz": -3.5}).build(),
            SpectrumBuilder().with_metadata({}).build(),
            SpectrumBuilder().with_metadata({"precursor_mz": 120.0}).build(),
        ]
    )

    filtered = require_precursor_mz(collection)

    assert filtered is not collection
    assert len(filtered) == 2
    assert filtered.metadata["precursor_mz"].tolist() == [60.0, 120.0]


def test_require_precursor_mz_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"precursor_mz": 60.0}).build(),
            SpectrumBuilder().with_metadata({"precursor_mz": 0.0}).build(),
        ]
    )

    filtered = require_precursor_mz(collection, clone=False)

    assert filtered is collection
    assert len(collection) == 1
    assert collection.metadata.loc[0, "precursor_mz"] == 60.0