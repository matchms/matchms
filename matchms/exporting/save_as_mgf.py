from typing import List
import pyteomics.mgf as py_mgf
from ..Spectrum import Spectrum
from ..utils import fingerprint_export_warning


def save_as_mgf(spectrums: List[Spectrum],
                filename: str,
                export_style: str = "matchms"):
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
    export_style:
        Converts the keys to the required export style. One of ["matchms", "massbank", "nist", "riken", "gnps"].
        Default is "matchms"
    """
    if not isinstance(spectrums, list):
        # Assume that input was single Spectrum
        spectrums = [spectrums]

    fingerprint_export_warning(spectrums)

    # Convert matchms.Spectrum() into dictionaries for pyteomics
    for spectrum in spectrums:
        spectrum_dict = {"m/z array": spectrum.peaks.mz,
                         "intensity array": spectrum.peaks.intensities,
                         "params": spectrum.metadata_dict(export_style)}
        if 'fingerprint' in spectrum_dict["params"]:
            del spectrum_dict["params"]["fingerprint"]
        # Append spectrum to file
        with open(filename, 'a', encoding="utf-8") as out:
            py_mgf.write(spectrum_dict, out)
