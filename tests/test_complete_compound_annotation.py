import numpy as np

from matchms import Spectrum
from matchms.filtering import complete_compound_annotation


def test_complete_compound_annotation_only_smiles():
    """Test if conversion to inchi and inchikey works when only smiles is given.
    """
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"smiles": "C1CCCCC1",
                                     "inchikey": 'n\a'})

    spectrum = complete_compound_annotation(spectrum_in)
    assert spectrum.get("inchi") == '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"', "Expected different InChI"
    assert spectrum.get("inchikey")[:14] == 'XDTMQSROBMDMFD', "Expected different inchikey"


def test_complete_compound_annotation_only_inchi():
    """Test if conversion to inchi and inchikey works when only smiles is given.
    """
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"inchi": '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"',
                                     "smiles": ""})

    spectrum = complete_compound_annotation(spectrum_in)
    assert spectrum.get("smiles") == "C1CCCCC1", "Expected different smiles"
    assert spectrum.get("inchikey")[:14] == 'XDTMQSROBMDMFD', "Expected different inchikey"


if __name__ == '__main__':
    test_complete_compound_annotation_only_smiles()
    test_complete_compound_annotation_only_inchi()
