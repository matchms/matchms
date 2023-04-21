import os
import pickle
from typing import Any, Generator, Optional
from matchms.importing import (load_from_json, load_from_mgf, load_from_msp,
                               load_from_mzml, load_from_mzxml, load_from_usi)
from matchms.typing import SpectrumType


def load_spectra(file: str, ftype: Optional[str] = None) -> Generator[SpectrumType, None, None]:
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
        return load_from_mzml(file)
    if ftype == "json":
        return load_from_json(file)
    if ftype == "mgf":
        return load_from_mgf(file)
    if ftype == "msp":
        return load_from_msp(file)
    if ftype == "mzxml":
        return load_from_mzxml(file)
    if ftype == "usi":
        return load_from_usi(file)
    if ftype == "pickle":
        spectra = load_from_pickle(file)
        assert isinstance(spectra, list), "Expected list of spectra"
        assert isinstance(spectra[0], SpectrumType), "Expected list of spectra"
        return spectra
    assert False, f"File extension of file: {file} is not recognized"


def load_from_pickle(filename: str) -> Any:
    """Load spectra stored in pickle

    Args:
        filename (str): Pickled file with spectra.

    Returns:
        Any: Unpickled object. Should be a list of Spectra.
    """
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object
