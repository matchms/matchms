import ast
import json
import numpy as np
from matchms import Spectrum


def load_from_json(filename):
    """Load spectrum(s) from json file.

    Args:
    ----
    filename: str
        Provide filename for json file containing spectrum(s).
    """
    not_metadata_fields = ["peaks_json"]
    parse_fieldnames = dict(inchi_aux="inchiaux",
                            ion_mode="ionmode")

    def parse_fieldname(key):
        """Add options to read GNPS style json files."""
        key_parsed = key.lower()
        key_parsed = parse_fieldnames.get(key_parsed, key_parsed)
        return key_parsed

    def get_peaks_list(spectrum_dict, fieldname):
        peaks_list = spectrum_dict.get(fieldname)
        if isinstance(peaks_list, list):
            return peaks_list
        # Handle peaks list when stored as string
        if isinstance(peaks_list, str):
            return ast.literal_eval(peaks_list)
        return []

    # Load from json file
    with open(filename, 'rb') as fin:
        spectrum_dicts = json.load(fin)

    spectrums = []
    for spectrum_dict in spectrum_dicts:

        metadata_dict = {parse_fieldname(key): spectrum_dict[key]
                         for key in spectrum_dict if key not in not_metadata_fields}
        peaks_list = get_peaks_list(spectrum_dict, "peaks_json")
        if len(peaks_list) > 0:
            spectrum = Spectrum(mz=np.array(peaks_list)[:, 0],
                                intensities=np.array(peaks_list)[:, 1],
                                metadata=metadata_dict)
            spectrums.append(spectrum)
        else:
            print("Empty spectrum found (no peaks in 'peaks_json').",
                  "Will not be imported.")

    return spectrums
