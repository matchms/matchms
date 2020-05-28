from typing import List
import pyteomics.mgf as py_mgf
from ..Spectrum import Spectrum


def save_as_mgf(spectrums: List[Spectrum], filename: str):
    """Save spectrum(s) as mgf file.

    :py:attr:`~matchms.Spectrum.losses` of spectrum will not be saved.

    Arguments:
    ----------
    spectrums:
        Expected input are match.Spectrum.Spectrum() objects.
    filename:
        Provide filename to save spectrum(s).
    """
    if not isinstance(spectrums, list):
        # Assume that input was single Spectrum
        spectrums = [spectrums]

    # Convert matchms.Spectrum() into dictionaries for pyteomics
    for spectrum in spectrums:
        spectrum_dict = {"m/z array": spectrum.peaks.mz,
                         "intensity array": spectrum.peaks.intensities,
                         "params": spectrum.metadata}
        # Append spectrum to file
        py_mgf.write([spectrum_dict], filename)
