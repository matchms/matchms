from typing import List, Union
import pyteomics.mgf as py_mgf
from matchms import Spectrum


def save_as_mgf(spectrums: Union[Spectrum, List[Spectrum]], filename: str):
    """Save spectrum(s) as mgf file.

    Args:
    ----
    spectrums: list of Spectrum() objects, Spectrum() object
        Expected input are match.Spectrum.Spectrum() objects.
    filename: str
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
