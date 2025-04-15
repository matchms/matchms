import logging
import os
from typing import IO, Dict, List, Union
from ..Fragments import Fragments
from ..Spectrum import Spectrum
from ..utils import (
    filter_empty_spectra,
    fingerprint_export_warning,
    load_export_key_conversions,
    rename_deprecated_params,
)


logger = logging.getLogger("matchms")
_extensions_not_allowed = ["mzml", "mzxml", "json", "mgf"]


@rename_deprecated_params(param_mapping={"spectrums": "spectra"}, version="0.26.5")
def save_as_msp(
    spectra: List[Spectrum],
    filename: str,
    write_peak_comments: bool = True,
    mode: str = "a",
    style: str = "matchms",
    peak_sep: str = '\t'
):
    """Save spectrum(s) as msp file.

    Example:

    .. code-block:: python

        import numpy as np
        from matchms import Spectrum
        from matchms.exporting import save_as_msp

        # Create dummy spectrum
        spectrum = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                            intensities=np.array([10, 10, 500], dtype="float"),
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
    write_peak_comments:
        Writes peak comments to individual peaks after the respective mz/intensity pair
        when set to True. Default is True.
    mode:
        Mode on how to write to file. One of ["w", "a"] (write/append). Default is append.
    style:
        Converts the keys to required Export style. One of ["massbank", "nist", "riken", "gnps"].
        Default is "matchms"
    peak_sep:
        Separator to use for writing the msp file.
    """
    # pylint: disable=too-many-arguments
    if not isinstance(spectra, list):
        # Assume that input was a single Spectrum.
        spectra = [spectra]

    spectra = filter_empty_spectra(spectra)
    fingerprint_export_warning(spectra)

    file_extension = filename.split(".")[-1]
    assert (
        file_extension.lower() not in _extensions_not_allowed
    ), f"File extension '.{file_extension}' not allowed."
    if not filename.endswith(".msp"):
        logger.warning(
            "Spectrum(s) will be stored as msp file with extension .%s",
            filename.split(".")[-1],
        )

    with open(filename, mode, encoding="utf-8") as outfile:
        for spectrum in spectra:
            _write_spectrum(spectrum, outfile, write_peak_comments, style, peak_sep)


def _write_spectrum(
    spectrum: Spectrum,
    outfile: IO,
    write_peak_comments: bool,
    export_style: str = "matchms",
    peak_sep: str = '\t'
):
    _write_metadata(spectrum, export_style, outfile)
    _write_peaks(
        spectrum.peaks,
        spectrum.peak_comments if write_peak_comments else None,
        outfile,
        peak_sep)
    outfile.write(os.linesep)


def _write_peaks(peaks: Fragments, peak_comments: Spectrum.peak_comments, outfile: IO, peak_sep: str):
    for mz, intensity in zip(peaks.mz, peaks.intensities):
        peak_comment = _format_peak_comment(mz, peak_comments, peak_sep)
        outfile.write(f"{mz}{peak_sep}{intensity}{peak_comment}\n")


def _write_metadata(spectrum: Spectrum, export_style: str, outfile: IO):
    metadata = spectrum.metadata_dict(export_style)
    key_conversions = load_export_key_conversions(export_style=export_style)

    metadata.pop(key_conversions['num_peaks'], None)
    metadata.pop('fingerprint', None)
    metadata.pop('peak_comments', None)

    compound_name = metadata.pop(key_conversions['compound_name'], None)
    if compound_name:
        outfile.write(f"{key_conversions['compound_name'].upper()}: {compound_name}\n")

    for key, value in metadata.items():
        if not (_is_num_peaks(key) or _is_peak_comments(key) or _is_fingerprint(key)):
            if key.upper().strip() == "SYNON: METB N": # Special case for GOLM
                for val in value:
                    outfile.write(f"{key.upper()}: {val}\n")
            else:
                outfile.write(f"{key.upper()}: {value}\n")
    outfile.write(f"NUM PEAKS: {len(spectrum.peaks)}\n")


def _format_peak_comment(mz: Union[int, float], peak_comments: Dict, peak_sep: str = '\t'):
    """Format peak comment for given mz to return the quoted comment or empty string if no peak comment is present."""
    if not isinstance(peak_comments, dict):
        return ""
    peak_comment = peak_comments.get(mz, None)
    if peak_comment is None:
        return ""
    return f'{peak_sep}"{peak_comment}"'


def _is_num_peaks(key: str) -> bool:
    return key.lower().startswith("num peaks") or key.lower().startswith("num_peaks")


def _is_peak_comments(key: str) -> bool:
    return key.lower().startswith("peak_comments")


def _is_fingerprint(key: str) -> bool:
    return key.lower().startswith("fingerprint")
