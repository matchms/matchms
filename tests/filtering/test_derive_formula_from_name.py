import pytest
from matchms.filtering import derive_formula_from_name
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, remove_formula_from_name, expected_formula, expected_compound_name",
    [
        [{"compound_name": "peptideXYZ [M+H+K] C5H12NO2"}, True, "C5H12NO2", "peptideXYZ [M+H+K]"],
        [
            {"compound_name": "peptideXYZ C5H12NO2", "formula": "totallycorrectformula"},
            True,
            "totallycorrectformula",
            "peptideXYZ",
        ],
        [{"compound_name": "peptideXYZ [M+H+K] C5H12NO2"}, False, "C5H12NO2", "peptideXYZ [M+H+K] C5H12NO2"],
        [{}, True, None, None],
    ],
)
def test_derive_formula_from_name(metadata, remove_formula_from_name, expected_formula, expected_compound_name):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = derive_formula_from_name(spectrum_in, remove_formula_from_name=remove_formula_from_name)

    assert spectrum.get("formula") == expected_formula, "Expected different formula."
    assert spectrum.get("compound_name") == expected_compound_name, "Expected different cleaned name."


@pytest.mark.parametrize(
    "string_addition, expected_formula",
    [
        ("C6H14NO2", "C6H14NO2"),
        ("C47H83N1O8P1", "C47H83N1O8P1"),
        ("HYPOTAURINE", None),
        ("CITRATE", None),
        ("NIST14", None),
        ("HCl", None),
        ("ACID", None),
        ("B12A13", None),
        ("(12)", None),
        ("6432", None),
        ("C15", None),
    ],
)
def test_derive_formula_from_name_examples(string_addition, expected_formula):
    spectrum_in = SpectrumBuilder().with_metadata({"compound_name": "peptideXYZ [M+H+K] " + string_addition}).build()

    spectrum = derive_formula_from_name(spectrum_in)

    assert spectrum.get("formula") == expected_formula, "Expected different formula."


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_formula_from_name(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
