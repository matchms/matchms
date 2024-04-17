import pytest
from matchms.filtering import require_retention_time
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, expected", [
    [{"retention_time": 7.163228}, SpectrumBuilder().with_metadata(
        {"retention_time": 7.163228}).build()],
    [{"compound_name": "Benfuracarb"}, None],
    [{}, None]
])
def test_require_retention_time(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_retention_time(spectrum_in)

    assert spectrum == expected, "Expected no changes."
