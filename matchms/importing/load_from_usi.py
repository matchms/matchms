import requests
import numpy as np
import json
from matchms import Spectrum


def load_from_usi(usi: str, server='https://metabolomics-usi.ucsd.edu'):
    """Load spectrum from metabolomics USI.

USI returns JSON data with keys 'peaks', 'n_peaks' and 'precuror_mz'

    Args:
    ----
    usi: str
        Provide the usi.

    server: string
        USI server
    """

    # Create the url
    url = server + '/json/?usi=' + usi
    metadata = {'usi': usi, 'server': server}
    response = requests.get(url)

    # Extract data and create Spectrum object
    try:
        spectral_data = response.json()
        peaks = spectral_data['peaks']
        mz_list, intensity_list = zip(*peaks)
        mz_array = np.array(mz_list)
        intensity_array = np.array(intensity_list)

        metadata['precursor_mz'] = spectral_data.get('precursor_mz', None)

        s = Spectrum(mz_array, intensity_array, metadata)

        return s
        
    except json.decoder.JSONDecodeError:
        # failed to unpack json
        return None
