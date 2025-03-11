import logging
import os
import pickle
from typing import List
from matchms.exporting import save_as_json, save_as_mgf, save_as_msp
from matchms.Spectrum import Spectrum
from matchms.utils import filter_empty_spectra, rename_deprecated_params


logger = logging.getLogger("matchms")


@rename_deprecated_params(param_mapping={"spectrums": "spectra"}, version="0.26.5")
def save_spectra(
    spectra: List[Spectrum],
    file: str,
    export_style: str = "matchms",
    append: bool = False,
) -> None:
    """Saves spectra as the file type specified.

    The following file extensions can be used: .json, .mgf, and .msp.

    Args:
    -----
    spectra:
        The spectra that are saved.
    file:
        Path to file containing spectra, with file extension ".json", ".mgf", ".msp".
    export_style:
        Converts the keys to the required export style.
        One of ["matchms", "massbank", "nist", "riken", "gnps"]. Default is "matchms".
    append:
        Only supported for ".mgf", and ".msp" filetypes. If True, will try to append
        to an existing file, instead of creating a new file. Default is `False`.
    """
    if os.path.exists(file) and not append:
        raise FileExistsError(f"The specified file: {file} already exists.")

    ftype = os.path.splitext(file)[1].lower()[1:]
    if append and ftype not in ("mgf", "msp"):
        raise ValueError(f"{ftype} isn't supported for when `append` is True")

    if not isinstance(spectra, list):
        spectra = [spectra]

    if len(spectra) == 0:
        logger.warning("No spectra to save. File will be empty.")
        with open(file, "w", encoding="utf-8"):
            pass
        return

    if ftype == "json":
        save_as_json(spectra, file, export_style)
    elif ftype == "mgf":
        save_as_mgf(spectra, file, export_style)
    elif ftype == "msp":
        save_as_msp(spectra, file, style=export_style, mode="a")
    elif ftype == "pickle":
        if export_style != "matchms":
            logger.error(
                "The only available export style for pickle is 'matchms', your export style %s",
                export_style,
            )
        save_as_pickled_file(spectra, file)
    else:
        raise TypeError(f"File extension of file: {file} is not recognized")


def save_as_pickled_file(spectra, filename: str) -> None:
    """Stores spectra as a pickled object

    Args:
    -----
    spectra:
        The spectra that are saved.
    filename:
        Path to file containing spectra, with file extension "json", "mgf", "msp"
    """
    if os.path.exists(filename):
        raise FileExistsError(f"The file '{filename}' already exists.")

    if not isinstance(spectra, list):
        raise TypeError("Expected list of spectra")
    if not isinstance(spectra[0], Spectrum):
        raise TypeError("Expected list of spectra")

    spectra = filter_empty_spectra(spectra)

    with open(filename, "wb") as f:
        pickle.dump(spectra, f)
