import numpy as np

from matchms import Spectrum
from matchms.filtering import derive_inchi_from_inchikey


def test_derive_inchi_from_inchikey():
    """Test if conversion from inchi and inchikey works."""
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"inchikey": 'n/a'})

    spectrum = derive_inchi_from_inchikey(spectrum_in)
    inchi = spectrum.get("inchi").replace('"', '')
    assert inchi == 'InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2', "Expected different InChI"


if __name__ == '__main__':
    test_derive_inchi_from_inchikey()
