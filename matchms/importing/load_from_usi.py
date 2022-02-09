import json
import logging
import numpy as np
import requests
from ..Spectrum import Spectrum


logger = logging.getLogger("matchms")


def load_from_usi(usi: str, server: str = "https://metabolomics-usi.ucsd.edu",
                  metadata_harmonization: bool = True):
    """Load spectrum from metabolomics USI.

    USI returns JSON data with keys "peaks", "n_peaks" and "precuror_mz"

    .. code-block:: python

        from matchms.importing import load_from_usi

        spectrum = load_from_usi("mzspec:MASSBANK::accession:SM858102")
        print(f"Found spectrum with precursor m/z of {spectrum.get("precursor_mz"):.2f}.")

    Parameters
    ----------
    usi:
        Provide the usi.
    server: string
        USI server
    metadata_harmonization : bool, optional
        Set to False if metadata harmonization to default keys is not desired.
        The default is True.
    """

    # Create the url
    url = server + "/json/?usi1=" + usi
    metadata = {"usi": usi, "server": server}
    response = requests.get(url)

    if response.status_code == 404:
        return None
    # Extract data and create Spectrum object
    try:
        spectral_data = response.json()
        if spectral_data is None or "peaks" not in spectral_data:
            logger.info("Empty spectrum found (no data found). Will not be imported.")
            return None
        peaks = spectral_data["peaks"]
        if len(peaks) == 0:
            logger.info("Empty spectrum found (no peaks in 'peaks_json'). Will not be imported.")
            return None
        mz_list, intensity_list = zip(*peaks)
        mz_array = np.array(mz_list)
        intensity_array = np.array(intensity_list)

        metadata["precursor_mz"] = spectral_data.get("precursor_mz", None)

        s = Spectrum(mz_array, intensity_array, metadata,
                     metadata_harmonization=metadata_harmonization)

        return s

    except json.decoder.JSONDecodeError:
        logger.warning("Failed to unpack json (JSONDecodeError).")
        return None
