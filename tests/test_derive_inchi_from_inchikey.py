import numpy
from matchms import Spectrum
from matchms.filtering import derive_inchikey_from_inchi


def test_derive_inchikey_from_inchi():
    """Test if conversion from inchi and inchikey works."""
    spectrum_in = Spectrum(mz=numpy.array([], dtype='float'),
                           intensities=numpy.array([], dtype='float'),
                           metadata={"inchi": '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"',
                                     "inchikey": 'n/a'})

    spectrum = derive_inchikey_from_inchi(spectrum_in)
    assert spectrum.get("inchikey")[:14] == 'XDTMQSROBMDMFD', "Expected different inchikey"
