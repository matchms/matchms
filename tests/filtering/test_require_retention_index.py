import pytest
from matchms import SpectraCollection
from matchms.filtering.metadata_processing.require_retention_index import require_retention_index
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"retention_index": 1817}, True),
    ],
)
def test_require_retention_index_with_retention_index_spectrum(metadata, expected):
    """Test if function correctly handles spectrum with retention index."""
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_retention_index(spectrum_in)

    assert (spectrum is not None) == expected, "Expected a Spectrum object."
    assert ("retention_index" in spectrum.metadata) == expected, "Expected the 'retention_index' key in the metadata."


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"retention_index": 1817}, True),
    ],
)
def test_require_retention_index_with_retention_index_collection(metadata, expected):
    """Test if function correctly handles collection row with retention index."""
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    collection = SpectraCollection([spectrum_in])

    filtered = require_retention_index(collection)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == int(expected)
    if expected:
        assert "retention_index" in filtered.metadata.columns


def test_require_retention_index_without_retention_index_spectrum():
    """Test if function correctly handles spectrum without retention index."""
    spectrum_in = SpectrumBuilder().with_metadata({}).build()

    spectrum = require_retention_index(spectrum_in)

    assert spectrum is None, "Expected None when retention index is missing."


def test_require_retention_index_without_retention_index_collection():
    """Test if function correctly drops collection row without retention index."""
    spectrum_in = SpectrumBuilder().with_metadata({}).build()
    collection = SpectraCollection([spectrum_in])

    filtered = require_retention_index(collection)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == 0


def test_require_retention_index_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"retention_index": 1817}).build(),
            SpectrumBuilder().with_metadata({}).build(),
            SpectrumBuilder().with_metadata({"retention_index": 2000}).build(),
        ]
    )

    filtered = require_retention_index(collection)

    assert filtered is not collection
    assert len(filtered) == 2
    assert filtered.metadata["retention_index"].tolist() == [1817, 2000]


def test_require_retention_index_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"retention_index": 1817}).build(),
            SpectrumBuilder().with_metadata({}).build(),
        ]
    )

    filtered = require_retention_index(collection, clone=False)

    assert filtered is collection
    assert len(collection) == 1
    assert collection.metadata.loc[0, "retention_index"] == 1817


def test_require_retention_index_with_input_none():
    """Test if function correctly handles None input."""
    spectrum = require_retention_index(None)

    assert spectrum is None, "Expected None when input is None."