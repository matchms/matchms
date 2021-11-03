from io import BytesIO
from typing import Generator, Union
import numpy
from pyteomics.mgf import MGF
from ..Spectrum import Spectrum


def load_from_mgf(source: Union[str, BytesIO]) -> Generator[Spectrum, None, None]:
    """Load spectrum(s) from mgf file.

    source param accepts both a path to a mgf file or a bytesIO object from a preloaded MGF file

    Example:

    .. code-block:: python

        from matchms.importing import load_from_mgf

        file_mgf = "pesticides.mgf"
        spectrums = list(load_from_mgf(file_mgf))

    """


    mgf = MGF(source, convert_arrays=1)

    for pyteomics_spectrum in mgf:

        metadata = pyteomics_spectrum.get("params", None)
        mz = pyteomics_spectrum["m/z array"]
        intensities = pyteomics_spectrum["intensity array"]

        # Sort by mz (if not sorted already)
        if not numpy.all(mz[:-1] <= mz[1:]):
            idx_sorted = numpy.argsort(mz)
            mz = mz[idx_sorted]
            intensities = intensities[idx_sorted]

        yield Spectrum(mz=mz, intensities=intensities, metadata=metadata)
