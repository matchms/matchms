import pytest
from matchms.filtering.derive_adduct_from_name import looks_like_adduct
from matchms.filtering.repair_adduct.clean_adduct import (_clean_adduct,
                                                          clean_adduct)
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


def test_looks_like_adduct():
    """Test if adducts are correctly identified"""
    for adduct in ["M+", "M*+", "M+Cl", "[M+H]", "[2M+Na]+", "M+H+K", "[2M+ACN+H]+",
                   "MS+Na", "MS+H", "M3Cl37+Na", "[M+H+H2O]"]:
        assert looks_like_adduct(adduct), "Expected this to be identified as adduct"
    for adduct in ["N+", "B*+", "++", "--", "[--]", "H+M+K"]:
        assert not looks_like_adduct(adduct), "Expected this not to be identified as adduct"
