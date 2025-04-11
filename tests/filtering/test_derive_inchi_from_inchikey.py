import pytest
from testfixtures import LogCapture
from matchms.filtering import derive_inchikey_from_inchi
from matchms.logging_functions import reset_matchms_logger, set_matchms_logger_level
from ..builder_Spectrum import SpectrumBuilder


def test_derive_inchikey_from_inchi():
    """Test if conversion from inchi and inchikey works."""
    pytest.importorskip("rdkit")
    set_matchms_logger_level("INFO")
    spectrum_in = SpectrumBuilder().with_metadata({"inchi": '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"', "inchikey": "n/a"}).build()

    with LogCapture() as log:
        spectrum = derive_inchikey_from_inchi(spectrum_in)
    assert spectrum.get("inchikey")[:14] == "XDTMQSROBMDMFD", "Expected different inchikey"
    log.check(("matchms", "INFO", "Added InChIKey XDTMQSROBMDMFD-UHFFFAOYSA-N to metadata (was converted from inchi)"))
    reset_matchms_logger()


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_inchikey_from_inchi(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
