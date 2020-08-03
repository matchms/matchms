from typing import Generator
from typing import Union
import numpy
from pyteomics import mzml
from matchms.Spectrum import Spectrum


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
        spectrums = [spec for spec in load_from_mzml(file_mzml)]

    Parameters
    ----------
    filename:
        Filename for mzml file to import.
    ms_level:
        Specify which ms level to import. Default is 2.
    """
    def parse_mzml_metadata(spec_mzml):
        """Parse relevant mzml metadata entries."""
        charge = None
        title = None
        precursor_mz = None
        scan_time = None

        first_search = list(find_by_key(spec_mzml, "precursor"))
        precursor_mz_search = next(find_by_key(first_search, "selected ion m/z"))
        if precursor_mz_search:
            precursor_mz = float(precursor_mz_search)
        precursor_charge = list(find_by_key(first_search, "charge state"))
        if precursor_charge:
            charge = int(precursor_charge[0]) 

        if "spectrum title" in spec_mzml:
            title = spec_mzml["spectrum title"]

        scan_time = next(find_by_key(spec_mzml, "scan start time"))

        return {"charge": charge,
                "title": title,
                "precursor_mz": precursor_mz,
                "scan start time": scan_time}

    for pyteomics_spectrum in list(mzml.read(filename, dtype=dict)):
        if "ms level" in pyteomics_spectrum and pyteomics_spectrum["ms level"] == ms_level:
            metadata = parse_mzml_metadata(pyteomics_spectrum)
            mz = numpy.asarray(pyteomics_spectrum["m/z array"], dtype="float")
            intensities = numpy.asarray(pyteomics_spectrum["intensity array"], dtype="float")

            if isinstance(mz, numpy.ndarray):
                # Sort by mz (if not sorted already)
                if not numpy.all(mz[:-1] <= mz[1:]):
                    idx_sorted = numpy.argsort(mz)
                    mz = mz[idx_sorted]
                    intensities = intensities[idx_sorted]

                yield Spectrum(mz=mz, intensities=intensities, metadata=metadata)
