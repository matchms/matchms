import logging
import os
from typing import IO, Dict, List, Union
from ..Fragments import Fragments
from ..Spectrum import Spectrum
from ..utils import filter_empty_spectra, fingerprint_export_warning, rename_deprecated_params


logger = logging.getLogger("matchms")
_extensions_not_allowed = ["mzml", "mzxml", "json", "mgf"]


@rename_deprecated_params(param_mapping={"spectrums": "spectra"}, version="0.26.5")
def save_as_msp(
    spectra: List[Spectrum],
    filename: str,
    write_peak_comments: bool = True,
    mode: str = "a",
    style: str = "matchms",
):
    """Save spectrum(s) as msp file.

    Example:

    .. code-block:: python

        import numpy as np
        from matchms import Spectrum
        from matchms.exporting import save_as_msp

        # Create dummy spectrum
        spectrum = Spectrum(
            mz=np.array([100, 200, 300], dtype="float"),
            intensities=np.array([10, 10, 500], dtype="float"),
            metadata={"charge": -1, "inchi": '"InChI=1S/C6H12"', "precursor_mz": 222.2},
        )

        # Write spectrum to test file
        save_as_msp(spectrum, "test.msp")

    Parameters
    ----------
    spectra:
        Expected input are match.Spectrum.Spectrum() objects.
    filename:
        Provide filename to save spectrum(s).
    write_peak_comments:
        Writes peak comments to individual peaks after the respective mz/intensity pair
        when set to True. Default is True.
    mode:
        Mode on how to write to file. One of ["w", "a"] (write/append). Default is append.
    style:
        Converts the keys to required Export style. One of ["massbank", "nist", "riken", "gnps"].
        Default is "matchms"
    """
    if not isinstance(spectra, list):
        # Assume that input was a single Spectrum.
        spectra = [spectra]

    spectra = filter_empty_spectra(spectra)
    fingerprint_export_warning(spectra)

    file_extension = filename.split(".")[-1]
    assert file_extension.lower() not in _extensions_not_allowed, f"File extension '.{file_extension}' not allowed."
    if not filename.endswith(".msp"):
        logger.warning(
            "Spectrum(s) will be stored as msp file with extension .%s",
            filename.split(".")[-1],
        )

    with open(filename, mode, encoding="utf-8") as outfile:
        for spectrum in spectra:
            _write_spectrum(spectrum, outfile, write_peak_comments, style)


def _write_spectrum(
    spectrum: Spectrum,
    outfile: IO,
    write_peak_comments: bool,
    export_style: str = "matchms",
):
    _write_metadata(spectrum.metadata_dict(export_style), outfile)
    if write_peak_comments is True:
        _write_peaks(spectrum.peaks, spectrum.peak_comments, outfile)
    else:
        _write_peaks(spectrum.peaks, None, outfile)
    outfile.write(os.linesep)


def _write_peaks(peaks: Fragments, peak_comments: Spectrum.peak_comments, outfile: IO):
    outfile.write(f"NUM PEAKS: {len(peaks)}\n")
    for mz, intensity in zip(peaks.mz, peaks.intensities):
        peak_comment = _format_peak_comment(mz, peak_comments)
        outfile.write(f"{mz}\t{intensity}{peak_comment}\n".expandtabs(12))


def _write_metadata(metadata: dict, outfile: IO):
    for key, value in metadata.items():
        if not (_is_num_peaks(key) or _is_peak_comments(key) or _is_fingerprint(key)):
            outfile.write(f"{key.upper()}: {value}\n")


def _format_peak_comment(mz: Union[int, float], peak_comments: Dict):
    """Format peak comment for given mz to return the quoted comment or empty string if no peak comment is present."""
    if not isinstance(peak_comments, dict):
        return ""
    peak_comment = peak_comments.get(mz, None)
    if peak_comment is None:
        return ""
    return f'\t"{peak_comment}"'


def _is_num_peaks(key: str) -> bool:
    return key.lower().startswith("num peaks") or key.lower().startswith("num_peaks")


def _is_peak_comments(key: str) -> bool:
    return key.lower().startswith("peak_comments")


def _is_fingerprint(key: str) -> bool:
    return key.lower().startswith("fingerprint")
