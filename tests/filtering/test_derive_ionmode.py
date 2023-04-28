from matchms.filtering import derive_ionmode
from ..builder_Spectrum import SpectrumBuilder


def test_derive_ionmode_positive_adduct():
    spectrum_in = SpectrumBuilder().with_metadata({"adduct": "[M+H]"}).build()

    spectrum = derive_ionmode(spectrum_in)

    assert spectrum.get("ionmode") == "positive", "Expected different ionmode."


def test_derive_ionmode_negative_adduct():
    spectrum_in = SpectrumBuilder().with_metadata({"adduct": "M-H-"}).build()

    spectrum = derive_ionmode(spectrum_in)

    assert spectrum.get("ionmode") == "negative", "Expected different ionmode."


def test_derive_ionmode_empty_spectrum():
    spectrum_in = None
    spectrum = derive_ionmode(spectrum_in)

    assert spectrum is None, "Expected differnt handling of None spectrum."
