import pytest
from matchms.filtering import derive_formula_from_inchi
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, overwrite, expected_formula", [
    [{"inchi": "InChI=1S/C15H21N3O4S/c1-23-7-6-12(15(21)22)16-13(19)9-18-14(20)8-10-4-2-3-5-11(10)17-18/h8,12H,2-7,9H2,1H3,(H,16,19)(H,21,22)/t12-/m0/s1"},
     True, "C15H21N3O4S"],
    [{"inchi": "InChI=1S/C15H21N3O4S/c1-23-7-6-12(15(21)22)16-13(19)9-18-14(20)8-10-4-2-3-5-11(10)17-18/h8,12H,2-7,9H2,1H3,(H,16,19)(H,21,22)/t12-/m0/s1"},
     False, "C15H21N3O4S"],
    [{"inchi": "InChI=1S/C15H21N3O4S/c1-23-7-6-12(15(21)22)16-13(19)9-18-14(20)8-10-4-2-3-5-11(10)17-18/h8,12H,2-7,9H2,1H3,(H,16,19)(H,21,22)/t12-/m0/s1",
     "formula": "wrong_formula"}, True, "C15H21N3O4S"],
    [{"inchi": "InChI=1S/C15H21N3O4S/c1-23-7-6-12(15(21)22)16-13(19)9-18-14(20)8-10-4-2-3-5-11(10)17-18/h8,12H,2-7,9H2,1H3,(H,16,19)(H,21,22)/t12-/m0/s1",
     "formula": "old_formula"}, False, "old_formula"],
    [{"inchi": "this is not really an inchi"},
     False, None],
    [{"inchi": "this is not really an inchi",
     "formula": "old_formula"}, True, "old_formula"],
    [{}, True, None],
])
def test_derive_formula_from_name(metadata, overwrite, expected_formula):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = derive_formula_from_inchi(spectrum_in, overwrite)
    assert spectrum.get("formula") == expected_formula, "Expected different formula."

