import numpy as np
import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.filtering import add_precursor_mz
from matchms.logging_functions import reset_matchms_logger, set_matchms_logger_level
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"precursor_mz": 444.0}, 444.0],
        [{}, None],
        [{"pepmass": (444.0, 10)}, 444.0],
    ],
)
def test_add_precursor_mz(metadata, expected, as_collection):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_precursor_mz,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("precursor_mz") == expected, "Expected different precursor_mz."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_add_precursor_mz_only_pepmass_present(caplog, as_collection):
    """Test if precursor_mz is correctly derived if only pepmass is present."""
    set_matchms_logger_level("INFO")
    mz = np.array([], dtype="float")
    intensities = np.array([], dtype="float")
    metadata = {"pepmass": (444.0, 10)}
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(metadata)
        .with_mz(mz)
        .with_intensities(intensities)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        add_precursor_mz,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("precursor_mz") == 444.0, "Expected different precursor_mz."
    assert "Added precursor_mz entry based on field 'pepmass'" in caplog.text
    reset_matchms_logger()


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "key, value, expected",
    [
        ["precursor_mz", "444.0", 444.0],
        ["precursormz", "15.6", 15.6],
        ["precursormz", 15.0, 15.0],
        ["precursor_mass", "17.887654", 17.887654],
        ["precursor_mass", "N/A", None],
        ["precursor_mass", "test", None],
        ["pepmass", (33.89, 50), 33.89],
        ["pepmass", "None", None],
        ["pepmass", None, None],
    ],
)
def test_add_precursor_mz_no_precursor_mz(key, value, expected, as_collection):
    """Test if precursor_mz is correctly derived from alternative fields."""
    mz = np.array([], dtype="float")
    intensities = np.array([], dtype="float")
    metadata = {key: value}
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(metadata)
        .with_mz(mz)
        .with_intensities(intensities)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        add_precursor_mz,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("precursor_mz") == expected, "Expected different precursor_mz."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "key, value, expected_log",
    [
        ["precursor_mz", "N/A", "No precursor_mz found in metadata."],
        ["precursor_mass", "test", "test can't be converted to float."],
        ["precursor_mz", None, "No precursor_mz found in metadata."],
        ["pepmass", None, "No precursor_mz found in metadata."],
        ["precursor_mz", [], "Found precursor_mz of undefined type."],
    ],
)
def test_add_precursor_mz_logging(key, value, expected_log, caplog, as_collection):
    """Test warning messages for invalid precursor_mz metadata."""
    mz = np.array([], dtype="float")
    intensities = np.array([], dtype="float")
    metadata = {key: value}
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(metadata)
        .with_mz(mz)
        .with_intensities(intensities)
        .build()
    )

    _ = run_filter_as_spectrum_or_collection(
        add_precursor_mz,
        spectrum_in,
        as_collection,
    )

    assert expected_log in caplog.text, "Expected different logging message."


def test_add_precursor_mz_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"precursormz": "15.6"}).build(),
            SpectrumBuilder().with_metadata({"precursor_mass": "N/A"}).build(),
            SpectrumBuilder().with_metadata({"pepmass": (33.89, 50)}).build(),
        ]
    )

    processed = add_precursor_mz(collection)

    assert processed is not collection
    assert processed.metadata.loc[0, "precursor_mz"] == pytest.approx(15.6)
    assert pd.isna(processed.metadata.loc[1, "precursor_mz"])
    assert processed.metadata.loc[2, "precursor_mz"] == pytest.approx(33.89)


def test_add_precursor_mz_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"precursormz": "15.6"}).build(),
        ]
    )

    processed = add_precursor_mz(collection, clone=False)

    assert processed is collection
    assert collection.metadata.loc[0, "precursor_mz"] == pytest.approx(15.6)


def test_add_precursor_mz_empty_spectrum():
    assert add_precursor_mz(None) is None