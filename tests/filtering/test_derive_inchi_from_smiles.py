import pytest
from matchms.filtering import derive_inchi_from_smiles
from ..builder_Spectrum import SpectrumBuilder


def test_derive_inchi_from_smiles():
    """Test if conversion to inchi works when only smiles is given."""
    pytest.importorskip("rdkit")
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": "C1CCCCC1"}).build()

    spectrum = derive_inchi_from_smiles(spectrum_in)
    inchi = spectrum.get("inchi").replace('"', "")
    assert inchi == "InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2", "Expected different InChI"


def test_derive_inchi_from_defect_smiles():
    """Test if conversion to inchi works when only smiles is given."""
    pytest.importorskip("rdkit")
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": "CX1CCCCC1"}).build()

    spectrum = derive_inchi_from_smiles(spectrum_in)
    inchi = spectrum.get("inchi", None)
    assert inchi is None, "Expected no InChI"


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_inchi_from_smiles(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
