import json
from typing import List
from ..Spectrum import Spectrum
from ..utils import filter_empty_spectra, fingerprint_export_warning, rename_deprecated_params


@rename_deprecated_params(param_mapping={"spectrums": "spectra"}, version="0.26.5")
def save_as_json(spectra: List[Spectrum], filename: str, export_style: str = "matchms"):
    """Save spectrum(s) as json file.

    Example:

    .. code-block:: python

        import numpy as np
        from matchms import Spectrum
        from matchms.exporting import save_as_json

        # Create dummy spectrum
        spectrum = Spectrum(
            mz=np.array([100, 200, 300], dtype="float"),
            intensities=np.array([10, 10, 500], dtype="float"),
            metadata={"charge": -1, "inchi": '"InChI=1S/C6H12"', "precursor_mz": 222.2},
        )

        # Write spectrum to test file
        save_as_json(spectrum, "test.json")

    Parameters
    ----------
    spectra:
        Expected input is a list of  :py:class:`~matchms.Spectrum.Spectrum` objects.
    filename:
        Provide filename to save spectrum(s).
    export_style:
        Converts the keys to the required export style. One of ["matchms", "massbank", "nist", "riken", "gnps"].
        Default is "matchms"
    """
    if not isinstance(spectra, list):
        # Assume that input was a single Spectrum.
        spectra = [spectra]

    spectra = filter_empty_spectra(spectra)
    fingerprint_export_warning(spectra)

    # Write to json file
    encoder_class = create_spectrum_json_encoder(export_style)
    with open(filename, "w", encoding="utf-8") as fout:
        json.dump(spectra, fout, cls=encoder_class)


def create_spectrum_json_encoder(export_style):
    class CustomSpectrumJSONEncoder(json.JSONEncoder):
        def default(self, o):
            """JSON Encoder for a matchms.Spectrum.Spectrum object"""
            if isinstance(o, Spectrum):
                spec = o.clone().to_dict(export_style)
                spec.pop("fingerprint", None)
                return spec
            return super().default(o)

    return CustomSpectrumJSONEncoder
