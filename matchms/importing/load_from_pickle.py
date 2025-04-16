import pickle
from typing import List
from matchms.Spectrum import Spectrum


def load_from_pickle(filename: str, metadata_harmonization: bool) -> List[Spectrum]:
    """Load spectra stored in pickle

    Args:
        filename (str): Pickled file with spectra.

    Returns:
        Any: Unpickled object. Should be a list of Spectra.
    """
    with open(filename, "rb") as file:
        loaded_object = pickle.load(file)

    if not isinstance(loaded_object, list):
        raise TypeError("Expected list of spectra")
    for spectrum in loaded_object:
        if not isinstance(spectrum, Spectrum):
            raise TypeError("Expected list of spectra")

    if metadata_harmonization:
        loaded_object = [
            Spectrum(spectrum.mz, spectrum.intensities, spectrum.metadata, metadata_harmonization)
            for spectrum in loaded_object
        ]
    return loaded_object
