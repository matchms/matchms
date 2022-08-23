from typing import Generator, TextIO, Union
import numpy as np
from pyteomics.mgf import MGF
from ..Spectrum import Spectrum


def load_from_mgf(source: Union[str, TextIO],
                  metadata_harmonization: bool = True) -> Generator[Spectrum, None, None]:
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
    metadata_harmonization : bool, optional
        Set to False if metadata harmonization to default keys is not desired.
        The default is True.
    """

    for pyteomics_spectrum in MGF(source, convert_arrays=1):

        metadata = pyteomics_spectrum.get("params", None)
        mz = pyteomics_spectrum["m/z array"]
        intensities = pyteomics_spectrum["intensity array"]

        # Sort by mz (if not sorted already)
        if not np.all(mz[:-1] <= mz[1:]):
            idx_sorted = np.argsort(mz)
            mz = mz[idx_sorted]
            intensities = intensities[idx_sorted]

        yield Spectrum(mz=mz,
                       intensities=intensities,
                       metadata=metadata,
                       metadata_harmonization=metadata_harmonization)
