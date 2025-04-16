import logging
from typing import List, Union
import pyteomics.mgf as py_mgf
from ..Spectrum import Spectrum
from ..utils import filter_empty_spectra, fingerprint_export_warning, rename_deprecated_params


logger = logging.getLogger("matchms")


@rename_deprecated_params(param_mapping={"spectrums": "spectra"}, version="0.26.5")
def save_as_mgf(
    spectra: Union[List[Spectrum], Spectrum],
    filename: str,
    export_style: str = "matchms",
    file_mode: str = "a",
) -> None:
    """Save spectrum(s) as mgf file.

    Example:

    .. code-block:: python

        import numpy as np
        from matchms import Spectrum
        from matchms.exporting import save_as_mgf

        # Create dummy spectrum
        spectrum = Spectrum(
            mz=np.array([100, 200, 300], dtype="float"),
            intensities=np.array([10, 10, 500], dtype="float"),
            metadata={"charge": -1, "inchi": '"InChI=1S/C6H12"', "precursor_mz": 222.2},
        )

        # Write spectrum to test file
        save_as_mgf(spectrum, "test.mgf")

    Parameters
    ----------
    spectra:
        Expected input are match.Spectrum.Spectrum() objects.
    filename:
        Provide filename to save spectrum(s).
    export_style:
        Converts the keys to the required export style. One of ["matchms", "massbank", "nist", "riken", "gnps"].
        Default is "matchms"
    file_mode: str
        This is an optional parameter which defines the mode the file will be opened in.
        Default is "a" (append). The other option is "w" (write). If set worngly it will default to "a".
    """
    if not isinstance(spectra, list):
        # Assume that input was a single Spectrum.
        spectra = [spectra]

    spectra = filter_empty_spectra(spectra)
    fingerprint_export_warning(spectra)

    def spectrum_dict_generator(matchms_spectra):
        """Generates dictionaries in the format expected by py_mgf"""
        for spectrum in matchms_spectra:
            spectrum_dict = {
                "m/z array": spectrum.peaks.mz,
                "intensity array": spectrum.peaks.intensities,
                "params": spectrum.metadata_dict(export_style),
            }
            if "fingerprint" in spectrum_dict["params"]:
                del spectrum_dict["params"]["fingerprint"]
            yield spectrum_dict

    if file_mode not in ["a", "w"]:
        file_mode = "a"
    py_mgf.write(spectrum_dict_generator(spectra), filename, file_mode=file_mode, encoding="utf-8")
