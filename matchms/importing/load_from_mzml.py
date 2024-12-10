from typing import Generator
import numpy as np
from pyteomics.mzml import read
from matchms.importing.parsing_utils import (parse_mzml_mzxml_metadata,
                                             sort_by_mz)
from matchms.Spectrum import Spectrum


def load_from_mzml(filename: str, ms_level: int = 2,
                   metadata_harmonization: bool = True) -> Generator[Spectrum, None, None]:
    """Load spectrum(s) from mzml file.

    This function will create ~matchms.Spectrum for every spectrum of desired
    ms_level found in a given MzML file. For more extensive parsing options consider
    using pyteomics or pymzml packages.

    Example:

    .. code-block:: python

        from matchms.importing import load_from_mzml

        file_mzml = "testdata.mzml"
        spectra = list(load_from_mzml(file_mzml))

    Parameters
    ----------
    filename:
        Filename for mzml file to import.
    ms_level:
        Specify which ms level to import. Default is 2.
    metadata_harmonization : bool, optional
        Set to False if metadata harmonization to default keys is not desired.
        The default is True.
    """
    with read(filename, dtype=dict) as reader:
        for pyteomics_spectrum in reader:
            if "ms level" in pyteomics_spectrum and pyteomics_spectrum["ms level"] == ms_level:
                mz = np.asarray(pyteomics_spectrum.pop("m/z array"), dtype="float")
                intensities = np.asarray(pyteomics_spectrum.pop("intensity array"), dtype="float")

                mz, intensities = sort_by_mz(mz=mz, intensities=intensities)
                parsed_metadata = parse_mzml_mzxml_metadata(pyteomics_spectrum)

                yield Spectrum(mz=mz, intensities=intensities, metadata=parsed_metadata,
                               metadata_harmonization=metadata_harmonization)
