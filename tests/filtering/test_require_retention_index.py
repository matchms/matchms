import pytest
from matchms.filtering import require_retention_index
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, expected", [
    [{"retention_index": 317}, SpectrumBuilder().with_metadata(
        {"retention_index": 317}).build()],
    [{"retention_index": 4135.446429}, SpectrumBuilder().with_metadata(
        {"retention_index": 4135.446429}).build()],
    [{"compound_name": "Acephate"}, None],
    [{}, None]
])
def test_require_retention_index(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_retention_index(spectrum_in)

    assert spectrum == expected, "Expected no changes."
