import pytest
from matchms.filtering import require_correct_ionmode
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "ionmode, ionmode_to_keep, spectrum_removed",
    [
        ("positive", "positive", False),
        ("negative", "negative", False),
        ("positive", "both", False),
        ("negative", "both", False),
        ("positive", "negative", True),
        ("negative", "positive", True),
        ("n/a", "both", True),
    ],
)
def test_require_correct_ionmode(ionmode, ionmode_to_keep, spectrum_removed):
    builder = SpectrumBuilder()
    spectrum_in = builder.with_metadata({"ionmode": ionmode}).build()
    spectrum = require_correct_ionmode(spectrum_in, ionmode_to_keep)

    if spectrum_removed is True:
        assert spectrum is None, "Expected spectrum to be filtered out since it does not have the correct ionmode"
    else:
        assert spectrum == spectrum_in
