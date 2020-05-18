import numpy
from matchms import Spectrum
from matchms.filtering import derive_adduct_from_name


def test_derive_adduct_from_name():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K]"})

    spectrum = derive_adduct_from_name(spectrum_in)

    assert spectrum.get("adduct") == "[M+H+K]", "Expected different adduct."
    assert spectrum.get("compound_name") == "peptideXYZ", "Expected different cleaned name."


def test_derive_adduct_from_name_dont_overwrite_present_adduct():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K]",
                                     "adduct": "M+H"})

    spectrum = derive_adduct_from_name(spectrum_in)

    assert spectrum.get("adduct") == "M+H", "Expected different adduct."
    assert spectrum.get("compound_name") == "peptideXYZ", "Expected different cleaned name."


def test_derive_adduct_from_name_dont_remove_from_name():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K]"})

    spectrum = derive_adduct_from_name(spectrum_in, remove_adduct_from_name=False)

    assert spectrum.get("adduct") == "[M+H+K]", "Expected different adduct."
    assert spectrum.get("compound_name") == spectrum_in.get("compound_name"), "Expected no change to name."
