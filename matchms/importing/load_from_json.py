import ast
import json
import logging
from typing import List, Union
import numpy as np
from ..Spectrum import Spectrum


logger = logging.getLogger("matchms")


def load_from_json(filename: str,
                   metadata_harmonization: bool = True) -> List[Spectrum]:
    """Load spectrum(s) from json file.

    JSON document formatted like the `GNPS Spectra library <https://gnps-external.ucsd.edu/gnpslibrary>`_.
    Spectrums with zero peaks will be skipped.

    Example:

    .. code-block:: python

        from matchms.importing import load_from_json

        file_json = "gnps_testdata.json"
        spectrums = load_from_json(file_json)

    Parameters
    ----------
    filename
        Provide filename for json file containing spectrum(s).
    metadata_harmonization : bool, optional
        Set to False if metadata harmonization to default keys is not desired.
        The default is True.
    """
    with open(filename, 'rb') as fin:
        spectrums = []
        for spectrum_dict in json.load(fin):
            spectrum = as_spectrum(spectrum_dict, metadata_harmonization=metadata_harmonization)
            if spectrum is not None:
                spectrums.append(spectrum)

    return spectrums


def as_spectrum(dct: dict,
                metadata_harmonization: bool = True) -> Union[dict, Spectrum, None]:
    """A :py:func:`json.load` object_hook to convert dictionary shaped like
    spectrum into :py:class:`~matchms.Spectrum.Spectrum` object.

    Parameters
    ----------
    dct
        Dictionary shaped like spectrum
    Returns
    -------
    A Spectrum or None when no peaks where found.
    """
    # Recognize Spectrum by peaks_json key
    if 'peaks_json' in dct:
        return dict2spectrum(dct, metadata_harmonization=metadata_harmonization)
    return None


def dict2spectrum(spectrum_dict: dict,
                  metadata_harmonization: bool) -> Union[Spectrum, None]:
    """Convert dictionary to a :py:class:`~matchms.Spectrum.Spectrum` object.

    Parameters
    ----------
    spectrum_dict
        Dictionary shaped like a single JSON object from the `GNPS Spectra library <https://gnps-external.ucsd.edu/gnpslibrary>`_

    Returns
    -------
    A Spectrum or None when no peaks where found.
    """
    not_metadata_fields = ["peaks_json"]
    parse_fieldnames = dict(inchi_aux="inchiaux",
                            ion_mode="ionmode")

    def get_peaks_list(spectrum_dict, fieldname):
        peaks_list = spectrum_dict.get(fieldname)
        if isinstance(peaks_list, list):
            return peaks_list
        # Handle peaks list when stored as string
        if isinstance(peaks_list, str):
            return ast.literal_eval(peaks_list)
        return []

    def parse_fieldname(key):
        """Add options to read GNPS style json files."""
        key_parsed = key.lower()
        key_parsed = parse_fieldnames.get(key_parsed, key_parsed)
        return key_parsed

    metadata_dict = {parse_fieldname(key): spectrum_dict[key]
                     for key in spectrum_dict if key not in not_metadata_fields}
    peaks_list = get_peaks_list(spectrum_dict, "peaks_json")
    if len(peaks_list) > 0 and metadata_dict:
        mz = np.array(peaks_list)[:, 0]
        intensities = np.array(peaks_list)[:, 1]

        # Sort by mz (if not sorted already)
        if not np.all(mz[:-1] <= mz[1:]):
            idx_sorted = np.argsort(mz)
            mz = mz[idx_sorted]
            intensities = intensities[idx_sorted]
        return Spectrum(mz=mz,
                        intensities=intensities,
                        metadata=metadata_dict,
                        metadata_harmonization=metadata_harmonization)
    logger.info("Empty spectrum found (no peaks in 'peaks_json'). Will not be imported.")
    return None


def scores_json_decoder(dct):
    """
    Object_hook function to convert JSON dictionary with :py:class:`~matchms.Score.Score` object into a python dictionary.
    """
    if "__Scores__" not in dct and "__Similarity__" not in dct:
        return dict2spectrum(dct, metadata_harmonization=False)
    return dct
