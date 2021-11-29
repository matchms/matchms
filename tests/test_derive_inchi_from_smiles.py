import numpy
import pytest
from testfixtures import LogCapture
from matchms import Spectrum
from matchms.filtering import derive_inchi_from_smiles


def test_derive_inchi_from_smiles():
    """Test if conversion to inchi works when only smiles is given.
    """
    pytest.importorskip("rdkit")
    spectrum_in = Spectrum(mz=numpy.array([], dtype='float'),
                           intensities=numpy.array([], dtype='float'),
                           metadata={"smiles": "C1CCCCC1"})

    spectrum = derive_inchi_from_smiles(spectrum_in)
    inchi = spectrum.get("inchi").replace('"', '')
    assert inchi == 'InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2', "Expected different InChI"


def test_derive_inchi_from_defect_smiles():
    """Test if conversion to inchi works when only smiles is given.
    """
    pytest.importorskip("rdkit")
    spectrum_in = Spectrum(mz=numpy.array([], dtype='float'),
                           intensities=numpy.array([], dtype='float'),
                           metadata={"smiles": "CX1CCCCC1"})

    with LogCapture() as log:
        spectrum = derive_inchi_from_smiles(spectrum_in)
    inchi = spectrum.get("inchi", None)
    assert inchi is None, "Expected no InChI"
    log.check(
        ('matchms', 'WARNING', "Could not convert smiles CX1CCCCC1 to InChI.")
    )


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_inchi_from_smiles(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
