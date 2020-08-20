import ast
import json
from typing import List
from typing import Union
import numpy
from ..Spectrum import Spectrum


def load_from_json(filename: str) -> List[Spectrum]:
    """Load spectrum(s) from json file.

    JSON document formatted like the `GNPS Spectra library <https://gnps-external.ucsd.edu/gnpslibrary>`_.
    Spectrums with zero peaks will be skipped.

    Example:

    .. code-block:: python

        from matchs.importing import load_from_json

        file_json = "gnps_testdata.json"
        spectrums = load_from_json(file_json)

    Parameters
    ----------
    filename
        Provide filename for json file containing spectrum(s).

    """
    with open(filename, 'rb') as fin:
        spectrums = []
        for spectrum_dict in json.load(fin):
            spectrum = as_spectrum(spectrum_dict)
            if spectrum is not None:
                spectrums.append(spectrum)

    return spectrums


def as_spectrum(dct: dict) -> Union[dict, Spectrum, None]:
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
        return dict2spectrum(dct)
    return None


def dict2spectrum(spectrum_dict: dict) -> Union[Spectrum, None]:
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
        mz = numpy.array(peaks_list)[:, 0]
        intensities = numpy.array(peaks_list)[:, 1]

        # Sort by mz (if not sorted already)
        if not numpy.all(mz[:-1] <= mz[1:]):
            idx_sorted = numpy.argsort(mz)
            mz = mz[idx_sorted]
            intensities = intensities[idx_sorted]
        return Spectrum(mz=mz,
                        intensities=intensities,
                        metadata=metadata_dict)
    print("Empty spectrum found (no peaks in 'peaks_json').",
          "Will not be imported.")
    return None
