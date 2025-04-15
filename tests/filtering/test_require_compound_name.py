import pytest
from matchms.filtering import require_compound_name
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"compound_name": "Acephate"}, SpectrumBuilder().with_metadata({"compound_name": "Acephate"}).build()],
        [{"formula": "H2O"}, None],
        [{}, None],
    ],
)
def test_require_compound_name(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_compound_name(spectrum_in)

    assert spectrum == expected, "Expected no changes."
