from pyteomics.mgf import MGF
from matchms.Spectrum import Spectrum

def load_from_mgf(filename):

    return [Spectrum(s) for s in list(MGF(filename, convert_arrays=1))]
