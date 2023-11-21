import pytest
from matchms.filtering import repair_not_matching_annotation
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("smile, inchi, inchikey, correct_inchikey", [
    ("CCC", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N"), #Already correct
    ("CCC", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWOOPSS-UHFFFAOYSA-N", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N"), #Typo
])
def test_repair_not_matching_annotation_repair_inchikey(smile, inchi, inchikey, correct_inchikey):
    """Checks if the inchikey is repaired, when smile and inchi already match"""
    builder = SpectrumBuilder()
    spectrum_in = builder.with_metadata({"smiles": smile,
                                         "inchi": inchi,
                                         "inchikey": inchikey}).build()
    spectrum_out = repair_not_matching_annotation(spectrum_in)
    assert spectrum_out.get("inchikey") == correct_inchikey
    assert spectrum_out.get("smiles") == spectrum_in.get("smiles")
    assert spectrum_out.get("inchi") == spectrum_in.get("inchi")


@pytest.mark.parametrize("smile, inchi, inchikey, parent_mass, correct_smiles, correct_inchi, correct_inchikey", [
    (None, "n/a", "wrong information", 100,
     None, "n/a", "wrong information"),  # no annotation and wrong field entries
    ("CCC", None, None, 100,
     "CCC", None, None),  # missing inchi and inchikey
    ("CCC", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N", 100,
     "CCC", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N"),  # Already correct
    ("CC", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWOOPSS-UHFFFAOYSA-N", 44,  # smiles and inchikey wrong
     "CCC", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N"),
    ("CCC", "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3", "ATUOYWHBWOOPSS-UHFFFAOYSA-N", 44,  # inchi wrong
     "CCC", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N"),
    ("C(=O)=O", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N", 44,  # Both match the parent mass
     "C(=O)=O", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N"),  # So nothing should be changed
    # Both don't match the parent mass, so nothing should be changed
    ("C(=O)=O", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N", 50,
     "C(=O)=O", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N"),
])
def test_repair_not_matching_annotation_repair_smiles_inchi(smile, inchi, inchikey, parent_mass,
                                                     correct_smiles, correct_inchi, correct_inchikey):
    # pylint: disable=too-many-arguments
    builder = SpectrumBuilder()
    spectrum_in = builder.with_metadata({"smiles": smile,
                                         "inchi": inchi,
                                         "inchikey": inchikey,
                                         "parent_mass": parent_mass}).build()
    spectrum_out = repair_not_matching_annotation(spectrum_in)
    assert spectrum_out.get("inchikey") == correct_inchikey
    assert spectrum_out.get("smiles") == correct_smiles
    assert spectrum_out.get("inchi") == correct_inchi
