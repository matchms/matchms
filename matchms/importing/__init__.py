"""
Functions for importing mass spectral data
##########################################

Matchms provides import functions for several commonly used mass spectral data
formats, such as *.mzML*, *.mzXML*, *.mgf*, or *.msp*. It is also possible to
load spectra from *.json* files (tested for json files from GNPS or json files
made with matchms), from pickle files, or based on a unique spectrum identifier
(USI) (:meth:`~matchms.importing.load_from_usi`).

The individual ``load_from_*`` functions and :meth:`~matchms.importing.load_spectra`
return spectra as :class:`~matchms.Spectrum.Spectrum` objects or iterables of
Spectrum objects. For collection-based workflows, use
:meth:`~matchms.importing.load_ms2_dataset` to directly load a file as a
:class:`~matchms.SpectraCollection.SpectraCollection`.

For more extensive import options we recommend building custom importers using
`pyteomics <https://github.com/levitsky/pyteomics>`_ or
`pymzml <https://github.com/pymzml/pymzML>`_.

To process spectrum metadata, matchms can also make use of known adduct
information which is imported via :mod:`~matchms.importing.load_adducts`.
"""

from .load_from_json import load_from_json
from .load_from_mgf import load_from_mgf
from .load_from_msp import load_from_msp
from .load_from_mzml import load_from_mzml
from .load_from_mzxml import load_from_mzxml
from .load_from_pickle import load_from_pickle
from .load_from_usi import load_from_usi
from .load_spectra import load_ms2_dataset, load_spectra


__all__ = [
    "load_from_json",
    "load_from_mgf",
    "load_from_msp",
    "load_from_mzml",
    "load_from_mzxml",
    "load_from_usi",
    "load_spectra",
    "load_from_pickle",
    "load_ms2_dataset",
]
