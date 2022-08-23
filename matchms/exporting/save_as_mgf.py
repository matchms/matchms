from typing import List
import pyteomics.mgf as py_mgf
from ..Spectrum import Spectrum


def save_as_mgf(spectrums: List[Spectrum], filename: str):
    """Save spectrum(s) as mgf file.

    :py:attr:`~matchms.Spectrum.losses` of spectrum will not be saved.

    Example:

    .. code-block:: python

        import numpy as np
        from matchms import Spectrum
        from matchms.exporting import save_as_mgf

        # Create dummy spectrum
        spectrum = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                            intensities=np.array([10, 10, 500], dtype="float"),
                            metadata={"charge": -1,
                                      "inchi": '"InChI=1S/C6H12"',
                                      "precursor_mz": 222.2})

        # Write spectrum to test file
        save_as_mgf(spectrum, "test.mgf")

    Parameters
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
