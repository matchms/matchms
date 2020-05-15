import numpy
from matchms import Spectrum
from matchms.filtering import add_adduct


def test_add_adduct_derived_from_name():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"name": "peptideXYZ [M+H+K]"})

    spectrum = add_adduct(spectrum_in)

    assert spectrum.get("adduct") == "[M+H+K]", "Expected different adduct."


def test_add_adduct_derived_from_compound_name():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K]"})

    spectrum = add_adduct(spectrum_in)

    assert spectrum.get("adduct") == "[M+H+K]", "Expected different adduct."


def test_add_adduct_dont_overwrite_present_adduct():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"name": "peptideXYZ [M+H+K]",
                                     "adduct": "M+"})

    spectrum = add_adduct(spectrum_in)

    assert spectrum.get("adduct") == "M+", "Expected different adduct."


def test_add_adduct_both_name_and_compound_name():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"name": "peptideX [M+H]",
                                     "compound_name": "peptideY M+H"})

    spectrum = add_adduct(spectrum_in)

    assert spectrum.get("adduct") == "[M+H]", "Expected adduct would be picked from name."
