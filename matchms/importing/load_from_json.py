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

    def parse_fieldname(key):
        """Add options to read GNPS style json files."""
        key_parsed = key.lower()
        return key_parsed

    # Load from json file
    with open(filename, 'rb') as fin:
        spectrum_dicts = json.load(fin)

    spectrums = []
    for spectrum_dict in spectrum_dicts:

        metadata_dict = {parse_fieldname(key): spectrum_dict[key]
                         for key in spectrum_dict if key not in not_metadata_fields}
        peaks_array = np.array(spectrum_dict.get("peaks_json"))
        spectrum = Spectrum(mz=peaks_array[:, 0],
                            intensities=peaks_array[:, 1],
                            metadata=metadata_dict)
        spectrums.append(spectrum)

    return spectrums
