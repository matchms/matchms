import pytest
from matchms.filtering import derive_smiles_from_inchi
from .builder_Spectrum import SpectrumBuilder


def test_derive_smiles_from_inchi():
    """Test if conversion to smiles works when only inchi is given.
    """
    pytest.importorskip("rdkit")
    spectrum_in = SpectrumBuilder().with_metadata(
        {"inchi": '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"',
         "smiles": ""}).build()

    spectrum = derive_smiles_from_inchi(spectrum_in)
    assert spectrum.get("smiles") == "C1CCCCC1", "Expected different smiles"


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_smiles_from_inchi(spectrum_in)

    assert spectrum is None, "Expected differnt handling of None spectrum."
