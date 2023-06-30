import pytest
from matchms.filtering.repair_adduct.clean_adduct import clean_adduct
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("input_adduct, expected_adduct",
                         [("M+", "[M]+"),
                          ("M+CH3COO-", "[M+CH3COO]-"),
                          ("M+CH3COO", "[M+CH3COO]-"),
                          ("M-CH3-", "[M-CH3]-"),
                          ("M+2H++", "[M+2H]2+"),
                          ("[2M+Na]", "[2M+Na]+"),
                          ("2M+Na", "[2M+Na]+"),
                          ("M+NH3+", "[M+NH3]+"),
                          ("M-H2O+2H2+", "[M-H2O+2H]2+"),
                          (None, None)])
def test_clean_adduct(input_adduct, expected_adduct):
    spectrum_in = SpectrumBuilder().with_metadata({"adduct": input_adduct}).build()
    spectrum_out = clean_adduct(spectrum_in)
    assert spectrum_out.get("adduct") == expected_adduct
