import logging
import os
from typing import IO
from typing import List
from ..Spectrum import Spectrum
from ..Spikes import Spikes


logger = logging.getLogger("matchms")


_extentions_not_allowed = ["mzml", "mzxml", "json", "mgf"]


def save_as_msp(spectra: List[Spectrum], filename: str):
    """Save spectrum(s) as msp file.

    :py:attr:`~matchms.Spectrum.losses` of spectrum will not be saved.

    Example:

    .. code-block:: python

        import numpy
        from matchms import Spectrum
        from matchms.exporting import save_as_msp

        # Create dummy spectrum
        spectrum = Spectrum(mz=numpy.array([100, 200, 300], dtype="float"),
                            intensities=numpy.array([10, 10, 500], dtype="float"),
                            metadata={"charge": -1,
                                      "inchi": '"InChI=1S/C6H12"',
                                      "precursor_mz": 222.2})

        # Write spectrum to test file
        save_as_msp(spectrum, "test.msp")

    Parameters
    ----------
    spectra:
        Expected input are match.Spectrum.Spectrum() objects.
    filename:
        Provide filename to save spectrum(s).
    """
    file_extension = filename.split(".")[-1]
    assert file_extension.lower() not in _extentions_not_allowed, \
        f"File extension '.{file_extension}' not allowed."
    if not filename.endswith(".msp"):
        logger.warning("Spectra will be stored as msp file with extension .%s",
                       filename.split(".")[-1])
    spectra = _ensure_list(spectra)

    with open(filename, "w", encoding="utf-8") as outfile:
        for spectrum in spectra:
            _write_spectrum(spectrum, outfile)


def _write_spectrum(spectrum: Spectrum, outfile: IO):
    _write_metadata(spectrum.metadata, outfile)
    _write_peaks(spectrum.peaks, outfile)
    outfile.write(os.linesep)


def _write_peaks(peaks: Spikes, outfile: IO):
    outfile.write(f"NUM PEAKS: {len(peaks)}\n")
    for mz, intensity in zip(peaks.mz, peaks.intensities):
        outfile.write(f"{mz}\t{intensity}\n")


def _write_metadata(metadata: dict, outfile: IO):
    for key, value in metadata.items():
        if not _is_num_peaks(key):
            outfile.write(f"{key.upper()}: {value}\n")


def _is_num_peaks(key: str) -> bool:
    return key.lower().startswith("num peaks")


def _ensure_list(spectra) -> List[Spectrum]:
    if not isinstance(spectra, list):
        # Assume that input was single Spectrum
        spectra = [spectra]
    return spectra
