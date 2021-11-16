import pytest
from matchms.filtering import derive_inchikey_from_inchi
from .builder_Spectrum import SpectrumBuilder


def test_derive_inchikey_from_inchi():
    """Test if conversion from inchi and inchikey works."""
    pytest.importorskip("rdkit")
    spectrum_in = SpectrumBuilder().with_metadata(
        {"inchi": '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"',
         "inchikey": 'n/a'}).build()

    spectrum = derive_inchikey_from_inchi(spectrum_in)
    assert spectrum.get("inchikey")[:14] == 'XDTMQSROBMDMFD', "Expected different inchikey"


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_inchikey_from_inchi(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
