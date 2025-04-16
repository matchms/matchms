import os
from pathlib import Path
from typing import Generator, TextIO, Union
from pyteomics.mgf import MGF
from matchms.importing.parsing_utils import parse_spectrum_dict
from matchms.Spectrum import Spectrum


def load_from_mgf(
    filename: Union[str, Path, TextIO], metadata_harmonization: bool = True
) -> Generator[Spectrum, None, None]:
    """Load spectrum(s) from mgf file.

    This function will create ~matchms.Spectrum for every spectrum in the given
    .mgf file (or the file-like object).

    Examples:

    .. code-block:: python

        from matchms.importing import load_from_mgf

        file_mgf = "pesticides.mgf"
        spectra_from_path = list(load_from_mgf(file_mgf))

        # Or you can read the file in your application
        with open(file_mgf, "r") as spectra_file:
            spectra_from_file = list(load_from_mgf(spectra_file))

    Parameters
    ----------
    filename:
        Accepts both filename (with path) for .mgf file or a file-like
        object from a preloaded MGF file.
    metadata_harmonization : bool, optional
        Set to False if metadata harmonization to default keys is not desired.
        The default is True.
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise FileNotFoundError(f"The specified file: {filename} doesn't exist.")

    if isinstance(filename, Path):
        filename = str(filename)  # pyteomics does not support pathlib.Path

    def parse_file():
        with MGF(filename, convert_arrays=1, encoding="utf-8") as reader:
            for pyteomics_spectrum in reader:
                yield parse_spectrum_dict(
                    spectrum=pyteomics_spectrum,
                    metadata_harmonization=metadata_harmonization,
                )

    return parse_file()
