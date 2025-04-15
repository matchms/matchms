import pytest
from matchms.filtering import require_formula
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"formula": "C6H12O6"}, SpectrumBuilder().with_metadata({"formula": "C6H12O6"}).build()],
        [{"formula": "Na2CO3"}, SpectrumBuilder().with_metadata({"formula": "Na2CO3"}).build()],
        [{"formula": "NaCl"}, SpectrumBuilder().with_metadata({"formula": "NaCl"}).build()],
        [{"formula": "H2O"}, SpectrumBuilder().with_metadata({"formula": "H2O"}).build()],
        [{"formula": "20C30H"}, None],
        [{}, None],
    ],
)
def test_require_formula(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_formula(spectrum_in)

    assert spectrum == expected, "Expected no changes."
