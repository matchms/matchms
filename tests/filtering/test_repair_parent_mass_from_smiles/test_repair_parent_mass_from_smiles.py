import math
import pytest
from matchms.filtering import repair_parent_mass_from_smiles
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "smiles, parent_mass, expected_parent_mass",
    [
        ("CN1CCCC1C2=CN=CC=C2", 162.23, 162.115698455),
        ("CN1CCCC1C2=CN=CC=C2", 162.1, 162.1),
    ],
)
def test_repair_parent_mass_from_smiles(smiles, parent_mass, expected_parent_mass):
    pytest.importorskip("rdkit")
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": smiles, "parent_mass": parent_mass}).build()
    spectrum_out = repair_parent_mass_from_smiles(spectrum_in, mass_tolerance=0.1)
    assert math.isclose(spectrum_out.get("parent_mass"), expected_parent_mass)
