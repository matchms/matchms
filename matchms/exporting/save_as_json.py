import copy
import json
from typing import List
from ..Spectrum import Spectrum


def save_as_json(spectrums: List[Spectrum], filename: str):
    """Save spectrum(s) as json file.

    :py:attr:`~matchms.Spectrum.losses` of spectrum will not be saved.

    Example:

    .. code-block:: python

        import numpy as np
        from matchms import Spectrum
        from matchms.exporting import save_as_json

        # Create dummy spectrum
        spectrum = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                            intensities=np.array([10, 10, 500], dtype="float"),
                            metadata={"charge": -1,
                                      "inchi": '"InChI=1S/C6H12"',
                                      "precursor_mz": 222.2})

        # Write spectrum to test file
        save_as_json(spectrum, "test.json")

    Parameters
    ----------
    spectrums:
        Expected input is a list of  :py:class:`~matchms.Spectrum.Spectrum` objects.
    filename:
        Provide filename to save spectrum(s).
    """
    if not isinstance(spectrums, list):
        # Assume that input was single Spectrum
        spectrums = [spectrums]

    # Write to json file
    with open(filename, "w", encoding="utf-8") as fout:
        json.dump(spectrums, fout, cls=SpectrumJSONEncoder)


class SpectrumJSONEncoder(json.JSONEncoder):
    # See https://github.com/PyCQA/pylint/issues/414 for reference
    def default(self, o):
        """JSON Encoder which can encode a :py:class:`~matchms.Spectrum.Spectrum` object"""
        if isinstance(o, Spectrum):
            spec = o.clone()
            return spec.to_dict()
        return json.JSONEncoder.default(self, o)


class ScoresJSONEncoder(json.JSONEncoder):
    def default(self, o):
        """JSON Encoder which can encode a :py:class:`~matchms.Scores.Scores` object"""
        class_name = o.__class__.__name__
        # do isinstance(o, Scores) without importing matchms.Scores
        if class_name == "Scores":
            scores = copy.deepcopy(o)
            return scores.to_dict()
        return json.JSONEncoder.default(self, o)
