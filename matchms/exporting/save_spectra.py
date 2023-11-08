import os
import pickle
from typing import List
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp, save_as_json
from matchms.typing import SpectrumType


def save_spectra(spectrums: List[Spectrum],
                 file: str) -> None:
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
    """
    assert os.path.exists(file), f"The specified file: {file} does not exists"

    ftype = os.path.splitext(file)[1].lower()[1:]

    if ftype == "json":
        save_as_json(spectrums, file)
    if ftype == "mgf":
        save_as_mgf(spectrums, file)
    if ftype == "msp":
        save_as_msp(spectrums, file)
    if ftype == "pickle":
        save_as_pickled_file(spectrums, file)

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

    if not isinstance(spectrums, list) or not isinstance(spectrums[0], SpectrumType):
        raise TypeError("Expected list of spectra")

    with open(filename, "wb") as f:
        pickle.dump(spectrums, f)
