from typing import Generator
from typing import Union
import numpy
from pyteomics import mzml
from matchms.Spectrum import Spectrum
from matchms.importing.parsing_utils import find_by_key
from matchms.importing.parsing_utils import parse_mzml_mzxml_metadata


def find_by_key(data: Union[list, dict], target: str):
    """Helper function to return entries from nested list/dictionary.

    Parameters
    ----------
    data:
        Nested dictionary or list in which entry should be searched.
    target:
        Name of field to search for in data.
    """
    if hasattr(data, "items"):
        for key, value in data.items():
            if isinstance(value, dict):
                if key == target:
                    yield value
                yield from find_by_key(value, target)
            elif isinstance(value, list):
                if key == target:
                    yield value
                for val in value:
                    yield from find_by_key(val, target)
            elif key == target:
                yield value

    elif isinstance(data, list):
        for subdata in data:
            yield from find_by_key(subdata, target)


def load_from_mzml(filename: str, ms_level: int = 2) -> Generator[Spectrum, None, None]:
    """Load spectrum(s) from mzml file.

    This function will create ~matchms.Spectrum for every spectrum of desired
    ms_level found in a given MzML file. For more extensive parsing options consider
    using pyteomics or pymzml packages.

    Example:

    .. code-block:: python

        from matchs.importing import load_from_mzml

        file_mzml = "testfile.mzml"
        spectrums = list(load_from_mzml(file_mzml))

    Parameters
    ----------
    filename:
        Filename for mzml file to import.
    ms_level:
        Specify which ms level to import. Default is 2.
    """
    for pyteomics_spectrum in list(mzml.read(filename, dtype=dict)):
        if "ms level" in pyteomics_spectrum and pyteomics_spectrum["ms level"] == ms_level:
            metadata = parse_mzml_mzxml_metadata(pyteomics_spectrum)
            mz = numpy.asarray(pyteomics_spectrum["m/z array"], dtype="float")
            intensities = numpy.asarray(pyteomics_spectrum["intensity array"], dtype="float")

            if isinstance(mz, numpy.ndarray):
                # Sort by mz (if not sorted already)
                if not numpy.all(mz[:-1] <= mz[1:]):
                    idx_sorted = numpy.argsort(mz)
                    mz = mz[idx_sorted]
                    intensities = intensities[idx_sorted]

                yield Spectrum(mz=mz, intensities=intensities, metadata=metadata)
