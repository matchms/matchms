import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import derive_formula_from_name


def test_derive_formula_from_name_default():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K] C5H12NO2"})

    spectrum = derive_formula_from_name(spectrum_in)

    assert spectrum.get("formula") == "C5H12NO2", "Expected different formula."
    assert spectrum.get("compound_name") == "peptideXYZ [M+H+K]", "Expected different cleaned name."


@pytest.mark.parametrize("string_addition, expected_formula", [("C6H14NO2", "C6H14NO2"),
                                                               ("C47H83N1O8P1", "C47H83N1O8P1"),
                                                               ("HYPOTAURINE", None),
                                                               ("CITRATE", None),
                                                               ("NIST14", None),
                                                               ("HCl", None),
                                                               ("ACID", None)])
def test_derive_formula_from_name_examples(string_addition, expected_formula):
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K] "+string_addition})

    spectrum = derive_formula_from_name(spectrum_in)

    assert spectrum.get("formula") == expected_formula, "Expected different formula."


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


def test_derive_formula_from_name_no_name_given():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={})

    spectrum = derive_formula_from_name(spectrum_in)

    assert spectrum.get("formula", None) is None, "Expected None for adduct."
    assert spectrum.get("compound_name", None) is None, "Expected None for name."


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_formula_from_name(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
