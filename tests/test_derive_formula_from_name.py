import numpy
from matchms import Spectrum
from matchms.filtering import derive_formula_from_name


def test_derive_formula_from_name():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K] C5H12NO2"})

    spectrum = derive_formula_from_name(spectrum_in)

    assert spectrum.get("formula") == "C5H12NO2", "Expected different formula."
    assert spectrum.get("compound_name") == "peptideXYZ [M+H+K]", "Expected different cleaned name."


def test_derive_formula_from_name_dont_overwrite_present_adduct():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ C5H12NO2",
                                     "formula": "totallycorrectformula"})

    spectrum = derive_formula_from_name(spectrum_in)

    assert spectrum.get("formula") == "totallycorrectformula", "Expected different adduct."
    assert spectrum.get("compound_name") == "peptideXYZ", "Expected different cleaned name."


def test_derive_formula_from_name_remove_formula_false():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K] C5H12NO2"})

    spectrum = derive_formula_from_name(spectrum_in, remove_formula_from_name=False)

    assert spectrum.get("formula") == "C5H12NO2", "Expected different formula."
    assert spectrum.get("compound_name") == spectrum_in.get("compound_name"), "Expected no name change."
