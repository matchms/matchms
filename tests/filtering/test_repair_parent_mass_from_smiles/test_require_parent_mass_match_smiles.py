import pytest
from matchms.filtering import require_parent_mass_match_smiles
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "smiles, parent_mass, correct",
    # first part is correct
    [
        ("C1=NC2=NC=NC(=C2N1)N", 135.054, True),
        ("C1=NC2=NC=NC(=C2N1)N", 150.054, False),
    ],
)
def test_repair_precursor_is_parent_mass(smiles, parent_mass, correct):
    pytest.importorskip("rdkit")

    spectrum_in = SpectrumBuilder().with_metadata({"smiles": smiles, "parent_mass": parent_mass}).build()
    spectrum_out = require_parent_mass_match_smiles(spectrum_in, mass_tolerance=0.1)
    if correct:
        assert spectrum_in == spectrum_out
    else:
        assert spectrum_out is None
