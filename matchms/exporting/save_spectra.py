import logging
import os
import pickle
from typing import List
from matchms.exporting import save_as_json, save_as_mgf, save_as_msp
from matchms.Spectrum import Spectrum


logger = logging.getLogger("matchms")


def save_spectra(spectrums: List[Spectrum],
                 file: str,
                 export_style: str = "matchms") -> None:
    """Saves spectra as the file type specified.

    The following file extensions can be used:
    "json", "mgf", "msp"

    Args:
    -----
    spectrums:
        The spectra that are saved.
    file:
        Path to file containing spectra, with file extension "json", "mgf", "msp"
    ftype:
        Optional. Filetype
    export_style:
        Converts the keys to the required export style. One of ["matchms", "massbank", "nist", "riken", "gnps"].
        Default is "matchms"
    """
    if os.path.exists(file):
        raise FileExistsError(f"The specified file: {file} already exists.")

    ftype = os.path.splitext(file)[1].lower()[1:]

    if ftype == "json":
        save_as_json(spectrums, file, export_style)
    elif ftype == "mgf":
        save_as_mgf(spectrums, file, export_style)
    elif ftype == "msp":
        save_as_msp(spectrums, file, style=export_style)
    elif ftype == "pickle":
        if export_style != "matchms":
            logger.error("The only available export style for pickle is 'matchms', your export style %s", export_style)
        save_as_pickled_file(spectrums, file)
    else:
        raise TypeError(f"File extension of file: {file} is not recognized")


def save_as_pickled_file(spectrums, filename: str) -> None:
    """Stores spectra as a pickled object

    Args:
    -----
    spectrums:
        The spectra that are saved.
    filename:
        Path to file containing spectra, with file extension "json", "mgf", "msp"
    """
    if os.path.exists(filename):
        raise FileExistsError(f"The file '{filename}' already exists.")

    if not isinstance(spectrums, list):
        raise TypeError("Expected list of spectra")
    if not isinstance(spectrums[0], Spectrum):
        raise TypeError("Expected list of spectra")

    with open(filename, "wb") as f:
        pickle.dump(spectrums, f)
