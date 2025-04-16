import pytest
from matchms.filtering import require_valid_annotation
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "smile, inchi, inchikey, correct",
    [
        ("CCC", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N", True),
        ("CCChello", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N", False),
        ("CCC", "InChI=1S/C3H8/c1-3-2/h3H2,1-2", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N", False),
        ("CCC", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "ATUOYWHBWRKTHZUHFFFAOYSA-N", False),
        ("CCC", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3", "FDHFJXKRMIVNCQ-RRNMINROSA-N", False),
        ("CCC", "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3", "ATUOYWHBWRKTHZ-UHFFFAOYSA-N", False),
    ],
)
def test_require_valid_annotation(smile, inchi, inchikey, correct):
    pytest.importorskip("rdkit")
    builder = SpectrumBuilder()
    spectrum_in = builder.with_metadata({"smiles": smile, "inchi": inchi, "inchikey": inchikey}).build()
    spectrum_out = require_valid_annotation(spectrum_in)
    if correct is True:
        assert spectrum_out == spectrum_in
    else:
        assert spectrum_out is None
