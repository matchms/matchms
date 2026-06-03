import pytest
from matchms import SpectraCollection
from matchms.filtering.metadata_processing.require_retention_time import require_retention_time
from ..builder_Spectrum import SpectrumBuilder


TEST_CASES = [
    ({"retention_time": 1817}, 0, 2000, True),
    ({"retention_time": 2100}, 0, 2000, False),
    ({"retention_time": 100}, 200, 2000, False),
]


@pytest.mark.parametrize(
    "metadata, minimum_rt, maximum_rt, expected",
    TEST_CASES,
)
def test_require_retention_time_within_range_spectrum(metadata, minimum_rt, maximum_rt, expected):
    """Test if function correctly handles spectrum with retention time."""
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_retention_time(spectrum_in, minimum_rt, maximum_rt)

    assert (spectrum is not None) == expected, "Expected a Spectrum object."


@pytest.mark.parametrize(
    "metadata, minimum_rt, maximum_rt, expected",
    TEST_CASES,
)
def test_require_retention_time_within_range_collection(metadata, minimum_rt, maximum_rt, expected):
    """Test if function correctly drops collection rows outside the retention time range."""
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    collection = SpectraCollection([spectrum_in])

    filtered = require_retention_time(collection, minimum_rt, maximum_rt)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == int(expected)


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"retention_time": 1817}, True),
        ({"retention_time": 500.12}, True),
        ({"retention_time": "500.12"}, False),
    ],
)
def test_require_retention_time_with_retention_time_spectrum(metadata, expected):
    """Test if function correctly handles spectrum with retention time."""
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_retention_time(spectrum_in)

    assert (spectrum is not None) == expected, "Expected a Spectrum object."
    if spectrum is not None:
        assert "retention_time" in spectrum.metadata


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"retention_time": 1817}, True),
        ({"retention_time": 500.12}, True),
        ({"retention_time": "500.12"}, False),
    ],
)
def test_require_retention_time_with_retention_time_collection(metadata, expected):
    """Test if function correctly handles collection rows with retention time."""
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    collection = SpectraCollection([spectrum_in])

    filtered = require_retention_time(collection)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == int(expected)

    if expected:
        assert "retention_time" in filtered.metadata.columns
        assert filtered.metadata.loc[0, "retention_time"] == metadata["retention_time"]


def test_require_retention_time_without_retention_time_spectrum():
    """Test if function correctly handles spectrum without retention time."""
    spectrum_in = SpectrumBuilder().with_metadata({}).build()

    spectrum = require_retention_time(spectrum_in)

    assert spectrum is None, "Expected None when retention time is missing."


def test_require_retention_time_without_retention_time_collection():
    """Test if function correctly drops collection rows without retention time."""
    spectrum_in = SpectrumBuilder().with_metadata({}).build()
    collection = SpectraCollection([spectrum_in])

    filtered = require_retention_time(collection)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == 0


def test_require_retention_time_collection_multiple_rows():
    """Test row-wise filtering of a mixed SpectraCollection."""
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"retention_time": 1817}).build(),
            SpectrumBuilder().with_metadata({"retention_time": 2100}).build(),
            SpectrumBuilder().with_metadata({"retention_time": 100}).build(),
            SpectrumBuilder().with_metadata({"retention_time": "500.12"}).build(),
            SpectrumBuilder().with_metadata({}).build(),
            SpectrumBuilder().with_metadata({"retention_time": 500.12}).build(),
        ]
    )

    filtered = require_retention_time(collection, minimum_rt=200, maximum_rt=2000)

    assert filtered is not collection
    assert len(filtered) == 2
    assert filtered.metadata["retention_time"].tolist() == [1817, 500.12]


def test_require_retention_time_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"retention_time": 1817}).build(),
            SpectrumBuilder().with_metadata({"retention_time": 2100}).build(),
        ]
    )

    filtered = require_retention_time(collection, minimum_rt=0, maximum_rt=2000, clone=False)

    assert filtered is collection
    assert len(collection) == 1
    assert collection.metadata.loc[0, "retention_time"] == 1817


def test_require_retention_time_with_input_none():
    """Test if function correctly handles None input."""
    spectrum = require_retention_time(None)

    assert spectrum is None, "Expected None when input is None."