import pytest
from matchms.filtering.metadata_processing.require_retention_time import \
    require_retention_time

from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, expected", [
    ({"retention_time": 1817}, True),
])
def test_require_retention_time_with_retention_time(metadata, expected):
    """Test if function correctly handles spectrum with retention time."""
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_retention_time(spectrum_in)

    assert (spectrum is not None) == expected, "Expected a Spectrum object."
    assert ("retention_time" in spectrum.metadata) == expected, "Expected the 'retention_time' key in the metadata."

@pytest.mark.parametrize("metadata, expected", [
    ({}, None),
])
def test_require_retention_time_without_retention_time(metadata, expected):
    """Test if function correctly handles spectrum without retention time."""
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_retention_time(spectrum_in)

    assert spectrum is expected, "Expected None when retention time is missing."

@pytest.mark.parametrize("spectrum_in, expected", [
    (None, None),
])
def test_require_retention_time_with_input_none(spectrum_in, expected):
    """Test if function correctly handles None input."""
    spectrum = require_retention_time(spectrum_in)

    assert spectrum is expected, "Expected None when input is None."
