import pytest
from matchms.filtering.metadata_processing.require_retention_index import require_retention_index
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"retention_index": 1817}, True),
    ],
)
def test_require_retention_index_with_retention_index(metadata, expected):
    """Test if function correctly handles spectrum with retention index."""
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_retention_index(spectrum_in)

    assert (spectrum is not None) == expected, "Expected a Spectrum object."
    assert ("retention_index" in spectrum.metadata) == expected, "Expected the 'retention_index' key in the metadata."


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({}, None),
    ],
)
def test_require_retention_index_without_retention_index(metadata, expected):
    """Test if function correctly handles spectrum without retention index."""
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_retention_index(spectrum_in)

    assert spectrum is expected, "Expected None when retention index is missing."


@pytest.mark.parametrize(
    "spectrum_in, expected",
    [
        (None, None),
    ],
)
def test_require_retention_index_with_input_none(spectrum_in, expected):
    """Test if function correctly handles None input."""
    spectrum = require_retention_index(spectrum_in)

    assert spectrum is expected, "Expected None when input is None."
