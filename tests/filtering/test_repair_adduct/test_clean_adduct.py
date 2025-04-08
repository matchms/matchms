import pytest
from matchms.filtering.metadata_processing.clean_adduct import _add_missing_brackets_to_adduct, _convert_int_charge_to_str, clean_adduct
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "input_adduct, expected_adduct",
    [
        ("M+", "[M]+"),
        ("M+CH3COO-", "[M+CH3COO]-"),
        ("M+CH3COO", "[M+CH3COO]-"),
        ("M-CH3-", "[M-CH3]-"),
        ("M+2H++", "[M+2H]2+"),
        ("[2M+Na]", "[2M+Na]+"),
        ("2M+Na", "[2M+Na]+"),
        ("M+NH3+", "[M+NH3]+"),
        ("M-H2O+2H2+", "[M-H2O+2H]2+"),
        (None, None),
    ],
)
def test_clean_adduct(input_adduct, expected_adduct):
    spectrum_in = SpectrumBuilder().with_metadata({"adduct": input_adduct}).build()
    spectrum_out = clean_adduct(spectrum_in)
    assert spectrum_out.get("adduct") == expected_adduct


@pytest.mark.parametrize(
    "input_charge, expected_charge",
    [
        (1, "+"),
        (2, "2+"),
        (-1, "-"),
        (+2, "2+"),
        (-2, "2-"),
        (None, None),
        ("None", None),
    ],
)
def test_convert_int_charge_to_str(input_charge, expected_charge):
    assert _convert_int_charge_to_str(input_charge) == expected_charge


@pytest.mark.parametrize(
    "input_adduct, expected_adduct",
    [
        ("M+", "[M]+"),
        ("M+CH3COO-", "[M+CH3COO]-"),
        ("M+CH3COO", "[M+CH3COO]"),
        ("M-CH3-", "[M-CH3]-"),
        ("[2M+Na]", "[2M+Na]"),
        ("2M+Na", "[2M+Na]"),
        ("M+NH3+", "[M+NH3]+"),
        ("M-H2O+2H2+", "[M-H2O+2H]2+"),
    ],
)
def test_add_missing_brackets_to_adduct(input_adduct, expected_adduct):
    assert _add_missing_brackets_to_adduct(input_adduct) == expected_adduct


@pytest.mark.parametrize(
    "input_adduct, charge, expected_adduct, expected_charge",
    [
        ("M", 1, "[M]+", 1),
        ("M", -1, "[M]-", -1),
        ("[M]+", None, "[M]+", 1),
        ("M+", None, "[M]+", 1),
        ("M-H2O+2H2-", None, "[M-H2O+2H]2-", -2),
        ("M-H2O+2H2+", None, "[M-H2O+2H]2+", 2),
        (None, 1, None, 1),
    ],
)
def test_clean_adduct_with_charge(input_adduct, charge, expected_adduct, expected_charge):
    spectrum_in = SpectrumBuilder().with_metadata({"adduct": input_adduct, "charge": charge}).build()
    spectrum_out = clean_adduct(spectrum_in)
    assert spectrum_out.get("adduct") == expected_adduct
    assert spectrum_out.get("charge") == expected_charge
