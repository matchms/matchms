import pytest
from matchms.filtering import derive_adduct_from_name
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, remove_adduct_from_name, expected_adduct, expected_name", [
    [{"compound_name": "peptideXYZ [M+H+K]"}, True, "[M+H+K]", "peptideXYZ"],
    [{"compound_name": "peptideXYZ [M+H+K]", "adduct": "M+H"}, True, "M+H", "peptideXYZ"],
    [{"compound_name": "peptideXYZ [M+H+K]"}, False, "[M+H+K]", "peptideXYZ [M+H+K]"],
    [{"name": ""}, True, None, None]
])
def test_derive_adduct_from_name_parametrized(metadata, remove_adduct_from_name, expected_adduct, expected_name):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = derive_adduct_from_name(spectrum_in, remove_adduct_from_name=remove_adduct_from_name)

    assert spectrum.get("adduct") == expected_adduct, "Expected different adduct."
    assert spectrum.get("compound_name") == expected_name, "Expected different cleaned name."


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_adduct_from_name(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
