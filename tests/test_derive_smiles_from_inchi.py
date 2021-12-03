import numpy
import pytest
from testfixtures import LogCapture
from matchms import Spectrum
from matchms.filtering import derive_smiles_from_inchi
from matchms.logging_functions import reset_matchms_logger
from matchms.logging_functions import set_matchms_logger_level


def test_derive_smiles_from_inchi():
    """Test if conversion to smiles works when only inchi is given.
    """
    pytest.importorskip("rdkit")
    set_matchms_logger_level("INFO")
    spectrum_in = Spectrum(mz=numpy.array([], dtype='float'),
                           intensities=numpy.array([], dtype='float'),
                           metadata={"inchi": '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"',
                                     "smiles": ""})

    with LogCapture() as log:
        spectrum = derive_smiles_from_inchi(spectrum_in)
    assert spectrum.get("smiles") == "C1CCCCC1", "Expected different smiles"
    log.check(
        ('matchms', 'INFO', 'Added smiles C1CCCCC1 to metadata (was converted from InChI)')
    )
    reset_matchms_logger()


def test_derive_smiles_from_defect_inchi():
    """Test if conversion to smiles works when only inchi is given.
    """
    pytest.importorskip("rdkit")
    spectrum_in = Spectrum(mz=numpy.array([], dtype='float'),
                           intensities=numpy.array([], dtype='float'),
                           metadata={"inchi": '"InChI=1S/C6H12/c1-2-XA4-6-5-3-1/h1-6H2"',
                                     "smiles": ""})

    spectrum = derive_smiles_from_inchi(spectrum_in)
    assert spectrum.get("smiles", None) == "", "Expected no smiles"


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_smiles_from_inchi(spectrum_in)

    assert spectrum is None, "Expected differnt handling of None spectrum."
