from typing import List
from pyteomics.mgf import MGF
from ..Spectrum import Spectrum


def load_from_mgf(filename: str) -> List[Spectrum]:
    """load from mgf file"""

    spectrums = list()
    for pyteomics_spectrum in MGF(filename, convert_arrays=1):

        metadata = pyteomics_spectrum.get("params", None)
        mz = pyteomics_spectrum["m/z array"]
        intensities = pyteomics_spectrum["intensity array"]

        spectrum = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
        spectrums.append(spectrum)

    return spectrums
