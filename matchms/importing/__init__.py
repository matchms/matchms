"""Functions for importing mass spectral data.

Matchms provides a import functions for several commonly used data types, such
as _.mzML_, _.mzXML_, _.mgf_, or _.msp_. It is also possible to load data from
_.json_ files (tested for json files from GNPS or json files made with matchms).
Another option is to load spectra based on a unique identifier (USI)
(:meth:`~matchms.simporting.load_from_usi`). For more extensive import options
we recommend building custom importers using __pyteomics__
(https://github.com/levitsky/pyteomics) or __pymzml__ (https://github.com/pymzml/pymzML).
"""
from .load_adducts import load_adducts
from .load_from_json import load_from_json
from .load_from_mgf import load_from_mgf
from .load_from_msp import load_from_msp
from .load_from_mzml import load_from_mzml
from .load_from_mzxml import load_from_mzxml
from .load_from_usi import load_from_usi


__all__ = [
    "load_from_json",
    "load_from_mgf",
    "load_from_msp",
    "load_from_mzml",
    "load_from_mzxml",
    "load_from_usi",
    "load_adducts"
]
