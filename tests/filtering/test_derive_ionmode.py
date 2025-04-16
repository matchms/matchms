import pytest
from matchms.filtering import derive_ionmode
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "adduct, charge, ionmode, expected_ionmode",
    [
        ["[M+H]", 1, None, "positive"],
        ["[M+H]", 1, "blabla", "positive"],
        [None, None, "blabla", "blabla"],
        ["M-H-", -1, None, "negative"],
        ["M+H", None, None, "positive"],
        ["blabla", 3, None, "positive"],
    ],
)
def test_derive_ionmode(adduct, charge, ionmode, expected_ionmode):
    spectrum_in = SpectrumBuilder().with_metadata({"adduct": adduct, "charge": charge, "ionmode": ionmode}).build()
    spectrum = derive_ionmode(spectrum_in)
    assert spectrum.get("ionmode") == expected_ionmode, "Expected different ionmode."


def test_derive_ionmode_empty_spectrum():
    spectrum_in = None
    spectrum = derive_ionmode(spectrum_in)

    assert spectrum is None, "Expected differnt handling of None spectrum."
