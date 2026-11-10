"""
Functions for exporting mass spectral data
##########################################

Individual :class:`~matchms.spectrum.Spectrum`, or lists of :class:`~matchms.spectrum.Spectrum`
can be exported to json, mgf, msp, mzML, or mzSpecLib files.
"""

from .save_as_json import save_as_json
from .save_as_mgf import save_as_mgf
from .save_as_msp import save_as_msp
from .save_as_mzml import save_as_mzml
from .save_as_mzspeclib import save_as_mzspeclib
from .save_as_parquet import save_as_parquet
from .save_spectra import SUPPORTED_FILE_FORMATS, save_spectra


__all__ = [
    "save_as_json",
    "save_as_mgf",
    "save_as_msp",
    "save_as_mzml",
    "save_as_mzspeclib",
    "save_as_parquet",
    "save_spectra",
]

SUPPORTED_FILE_FORMATS = {**SUPPORTED_FILE_FORMATS, "mzspeclib": "mzSpecLib"}
