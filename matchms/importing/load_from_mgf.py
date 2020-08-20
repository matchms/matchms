from typing import Generator
import numpy
from pyteomics.mgf import MGF
from ..Spectrum import Spectrum


def load_from_mgf(filename: str) -> Generator[Spectrum, None, None]:
    """Load spectrum(s) from mgf file.

    Example:

    .. code-block:: python

        from matchs.importing import load_from_mgf

        file_mgf = "pesticides.mgf"
        spectrums = list(load_from_mgf(file_mgf))

    """

    for pyteomics_spectrum in MGF(filename, convert_arrays=1):

        metadata = pyteomics_spectrum.get("params", None)
        mz = pyteomics_spectrum["m/z array"]
        intensities = pyteomics_spectrum["intensity array"]

        # Sort by mz (if not sorted already)
        if not numpy.all(mz[:-1] <= mz[1:]):
            idx_sorted = numpy.argsort(mz)
            mz = mz[idx_sorted]
            intensities = intensities[idx_sorted]

        yield Spectrum(mz=mz, intensities=intensities, metadata=metadata)
