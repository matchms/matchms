import os
import pickle
from typing import Generator, Optional
from matchms.importing import (load_from_json, load_from_mgf, load_from_msp,
                               load_from_mzml, load_from_mzxml, load_from_usi)
from matchms.typing import SpectrumType


def load_spectra(file: str, metadata_harmonization: bool = True, ftype: Optional[str] = None) -> Generator[SpectrumType, None, None]:
    """Loads spectra from your spectrum file into memory as matchms Spectrum object

    The following file extensions can be loaded in with this function:
    "mzML", "json", "mgf", "msp", "mzxml", "usi" and "pickle".
    A pickled file is expected to directly contain a list of matchms spectrum objects.

    Args:
    -----
    file:
        Path to file containing spectra, with file extension "mzML", "json", "mgf", "msp",
        "mzxml", "usi" or "pickle"
    ftype:
        Optional. Filetype
    """
    assert os.path.exists(file), f"The specified file: {file} does not exists"

    if ftype is None:
        ftype = os.path.splitext(file)[1].lower()[1:]
    else:
        ftype = ftype.lower()

    if ftype == "mzml":
        return load_from_mzml(file, metadata_harmonization=metadata_harmonization)
    if ftype == "json":
        return load_from_json(file, metadata_harmonization=metadata_harmonization)
    if ftype == "mgf":
        return load_from_mgf(file, metadata_harmonization=metadata_harmonization)
    if ftype == "msp":
        return load_from_msp(file, metadata_harmonization=metadata_harmonization)
    if ftype == "mzxml":
        return load_from_mzxml(file, metadata_harmonization=metadata_harmonization)
    if ftype == "usi":
        return load_from_usi(file, metadata_harmonization=metadata_harmonization)
    if ftype == "pickle":
        return load_from_pickle(file, metadata_harmonization)
        
    raise TypeError(f"File extension of file: {file} is not recognized")


def load_from_pickle(filename: str, metadata_harmonization: bool) -> SpectrumType:
    """Load spectra stored in pickle

    Args:
        filename (str): Pickled file with spectra.

    Returns:
        Any: Unpickled object. Should be a list of Spectra.
    """
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)

    if not isinstance(loaded_object, list) or not isinstance(loaded_object[0], SpectrumType):
        raise TypeError("Expected list of spectra")

    if metadata_harmonization:
        loaded_object = [SpectrumType(x.peaks.mz, x.peaks.intensisites, x.metadata, metadata_harmonization) for x in loaded_object] 
    return loaded_object
