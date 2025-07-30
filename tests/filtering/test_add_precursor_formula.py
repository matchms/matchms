import pytest
from matchms.filtering import add_precursor_formula
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected_formula",
    [
        [{"formula": "C12H15ClN2O3", "adduct": "[M+H]+"}, "C12H16ClN2O3"],
        [{"formula": "C12H15ClN2O3", "adduct": "[M+H-H2O]+"}, "C12H14ClN2O2"], # multiple adducts
        [{"formula": "C12H15ClN2O3"}, None],  # no adduct
        [{"adduct": "[M+H-H2O]+"}, None],  # no formula
        [{"formula": "C2H8NO", "adduct": "[2M+H-H2O]+"}, "C4H15N2O"],  # multiple masses
        [{"formula": "C2H6", "adduct": "[M-H2O]+"}, None],  # impossible adduct formula combo
        [{}, None],
    ],
)
def test_derive_formula_from_smiles(metadata, expected_formula):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = add_precursor_formula(spectrum_in)
    assert spectrum.get("precursor_formula") == expected_formula, "Expected different formula."
