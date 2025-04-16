import pytest
from matchms.filtering import repair_smiles_of_salts
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "smiles, parent_mass, expected_smiles, expected_salt_ions",
    # first part is correct
    [
        ("C1=NC2=NC=NC(=C2N1)N.Cl", 135.054, "C1=NC2=NC=NC(=C2N1)N", "Cl"),
        # Last part is correct
        ("C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O", 192.027, "C(C(=O)O)C(CC(=O)O)(C(=O)O)O", "C(C(=O)O)"),
        # Not a salt
        ("C(C(=O)O)C(CC(=O)O)(C(=O)O)O", 192.027, "C(C(=O)O)C(CC(=O)O)(C(=O)O)O", None),
        # All parts are incorrect
        ("C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O", 150.0, "C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O", None),
        # Salt with > 3 parts
        ("C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O.Cl", 192.027, "C(C(=O)O)C(CC(=O)O)(C(=O)O)O", "C(C(=O)O).Cl"),
        # Salt matching a combination of 2 parts
        ("C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O.Cl", 228.0, "C(C(=O)O)C(CC(=O)O)(C(=O)O)O.Cl", "C(C(=O)O)"),
    ],
)
def test_repair_precursor_is_parent_mass(smiles, parent_mass, expected_smiles, expected_salt_ions):
    pytest.importorskip("rdkit")

    spectrum_in = SpectrumBuilder().with_metadata({"smiles": smiles, "parent_mass": parent_mass}).build()
    spectrum_out = repair_smiles_of_salts(spectrum_in, mass_tolerance=0.1)
    assert spectrum_out.get("smiles") == expected_smiles
    assert spectrum_out.get("salt_ions") == expected_salt_ions
