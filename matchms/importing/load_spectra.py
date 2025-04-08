import os
from typing import Generator, List, Optional, Union
from matchms.importing import load_from_json, load_from_mgf, load_from_msp, load_from_mzml, load_from_mzxml, load_from_pickle, load_from_usi
from matchms.Spectrum import Spectrum


def load_spectra(file: str, metadata_harmonization: bool = True, ftype: Optional[str] = None) -> Union[List[Spectrum], Generator[Spectrum, None, None]]:
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


def load_list_of_spectrum_files(spectrum_files: Union[List[str], str]) -> Union[List[Spectrum], Generator[Spectrum, None, None]]:
    """Combines all spectra in multiple files into a list of spectra"""
    # Just load spectra if it is a single file
    if isinstance(spectrum_files, str):
        return load_spectra(spectrum_files)
    # If multiple files combine results into one generator
    spectrum_generators = [load_spectra(spectrum_file) for spectrum_file in spectrum_files]

    def chain(*iterables):
        """Combines multiple iterators (and generators) into a single iterator"""
        # chain('ABC', 'DEF') --> A B C D E F
        for it in iterables:
            for element in it:
                yield element

    return chain(*spectrum_generators)
