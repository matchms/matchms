import pytest
from matchms.filtering.metadata_processing.derive_formula_from_smiles import derive_formula_from_smiles
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, overwrite, expected_formula",
    [
        [{"smiles": "Cl.O.O.OC=1C=CC2=C(C1)NC3=C2C=CN=C3C"}, True, "C12H15ClN2O3"],
        [{"smiles": "Cl.O.O.OC=1C=CC2=C(C1)NC3=C2C=CN=C3C"}, False, "C12H15ClN2O3"],
        [{"smiles": "Cl.O.O.OC=1C=CC2=C(C1)NC3=C2C=CN=C3C", "formula": "wrong_formula"}, True, "C12H15ClN2O3"],
        [{"smiles": "Cl.O.O.OC=1C=CC2=C(C1)NC3=C2C=CN=C3C", "formula": "old_formula"}, False, "old_formula"],
        [{"smiles": "this is not really an inchi"}, False, None],
        [{"smiles": "this is not really an inchi", "formula": "old_formula"}, True, "old_formula"],
        [{}, True, None],
    ],
)
def test_derive_formula_from_smiles(metadata, overwrite, expected_formula):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = derive_formula_from_smiles(spectrum_in, overwrite)
    assert spectrum.get("formula") == expected_formula, "Expected different formula."
