"""
Functions for exporting mass spectral data
##########################################

Individual :meth:`~matchms.Spectrum`, or lists of :meth:`~matchms.Spectrum`
can be exported to json or mgf files.
"""
from .save_as_json import save_as_json
from .save_as_mgf import save_as_mgf


__all__ = [
    "save_as_json",
    "save_as_mgf",
]
