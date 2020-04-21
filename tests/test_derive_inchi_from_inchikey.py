import numpy as np

from matchms import Spectrum
from matchms.filtering import derive_inchi_from_inchikey


def test_derive_inchi_from_inchikey():
    """Test if conversion from inchi and inchikey works."""
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"inchikey": 'n/a'})

    spectrum = derive_inchi_from_inchikey(spectrum_in)
    assert spectrum.get("inchikey")[:14] == 'XDTMQSROBMDMFD', "Expected different inchikey"


if __name__ == '__main__':
    test_derive_inchi_from_inchikey()
