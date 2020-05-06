import numpy as np

from matchms import Spectrum
from matchms.filtering import derive_smiles_from_inchi


def test_derive_smiles_from_inchi():
    """Test if conversion to smiles works when only inchi is given.
    """
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"inchi": '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"',
                                     "smiles": ""})

    spectrum = derive_smiles_from_inchi(spectrum_in)
    assert spectrum.get("smiles") == "C1CCCCC1", "Expected different smiles"
