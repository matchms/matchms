from typing import Generator
from typing import TextIO
from typing import Union
import numpy
from pyteomics.mgf import MGF
from ..Spectrum import Spectrum


def load_from_mgf(source: Union[str, TextIO]) -> Generator[Spectrum, None, None]:
    """Load spectrum(s) from mgf file.

    This function will create ~matchms.Spectrum for every spectrum in the given
    .mgf file (or the file-like object).

    Examples:

    .. code-block:: python

        from matchms.importing import load_from_mgf

        file_mgf = "pesticides.mgf"
        spectra_from_path = list(load_from_mgf(file_mgf))

        # Or you can read the file in your application
        with open(file_mgf, 'r') as spectra_file:
            spectra_from_file = list(load_from_mgf(spectra_file))

    Parameters
    ----------
    source:
        Accepts both filename (with path) for .mgf file or a file-like
        object from a preloaded MGF file.
    """

    for pyteomics_spectrum in MGF(source, convert_arrays=1):

        metadata = pyteomics_spectrum.get("params", None)
        mz = pyteomics_spectrum["m/z array"]
        intensities = pyteomics_spectrum["intensity array"]

        # Sort by mz (if not sorted already)
        if not numpy.all(mz[:-1] <= mz[1:]):
            idx_sorted = numpy.argsort(mz)
            mz = mz[idx_sorted]
            intensities = intensities[idx_sorted]

        yield Spectrum(mz=mz, intensities=intensities, metadata=metadata)
