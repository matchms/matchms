from typing import Generator
import numpy as np
from pyteomics import mzxml
from matchms.importing.parsing_utils import parse_mzml_mzxml_metadata
from matchms.Spectrum import Spectrum


def load_from_mzxml(filename: str, ms_level: int = 2,
                    metadata_harmonization: bool = True) -> Generator[Spectrum, None, None]:
    """Load spectrum(s) from mzml file.

    This function will create ~matchms.Spectrum for every spectrum of desired
    ms_level found in a given MzXML file. For more extensive parsing options consider
    using the pyteomics package.

    Example:

    .. code-block:: python

        from matchms.importing import load_from_mzxml

        file_mzxml = "testdata.mzxml"
        spectrums = list(load_from_mzml(file_mzxml))

    Parameters
    ----------
    filename:
        Filename for mzXML file to import.
    ms_level:
        Specify which ms level to import. Default is 2.
    metadata_harmonization : bool, optional
        Set to False if metadata harmonization to default keys is not desired.
        The default is True.
    """
    for pyteomics_spectrum in mzxml.read(filename, dtype=dict):
        if ("ms level" in pyteomics_spectrum and pyteomics_spectrum["ms level"] == ms_level
                or "msLevel" in pyteomics_spectrum and pyteomics_spectrum["msLevel"] == ms_level):
            metadata = parse_mzml_mzxml_metadata(pyteomics_spectrum)
            mz = np.asarray(pyteomics_spectrum["m/z array"], dtype="float")
            intensities = np.asarray(pyteomics_spectrum["intensity array"], dtype="float")

            if mz.shape[0] > 0:
                # Sort by mz (if not sorted already)
                if not np.all(mz[:-1] <= mz[1:]):
                    idx_sorted = np.argsort(mz)
                    mz = mz[idx_sorted]
                    intensities = intensities[idx_sorted]

                yield Spectrum(mz=mz, intensities=intensities, metadata=metadata,
                               metadata_harmonization=metadata_harmonization)
