"""
Functions for importing mass spectral data
##########################################

Matchms provides a import functions for several commonly used data types, such
as *.mzML*, *.mzXML*, *.mgf*, or *.msp*. It is also possible to load data from
*.json* files (tested for json files from GNPS or json files made with matchms).
Another option is to load spectra based on a unique identifier (USI)
(:meth:`~matchms.importing.load_from_usi`).

For more extensive import options we recommend building custom importers using `pyteomics
<https://github.com/levitsky/pyteomics>`_ or `pymzml <https://github.com/pymzml/pymzML>`_.

To process spectrum metadata, matchms can also make use of known adduct information
which is imported via :mod:`~matchms.importing.load_adducts`.
"""
from .load_from_json import load_from_json
from .load_from_mgf import load_from_mgf
from .load_from_msp import load_from_msp
from .load_from_mzml import load_from_mzml
from .load_from_mzspeclib import load_from_mzspeclib
from .load_from_mzxml import load_from_mzxml
from .load_from_pickle import load_from_pickle
from .load_from_usi import load_from_usi
from .load_scores import scores_from_json, scores_from_pickle
from .load_spectra import load_spectra


__all__ = [
    "load_from_json",
    "load_from_mgf",
    "load_from_msp",
    "load_from_mzml",
    "load_from_mzspeclib",
    "load_from_mzxml",
    "load_from_usi",
    "load_spectra",
    "load_from_pickle",
    "scores_from_json",
    "scores_from_pickle",
]
