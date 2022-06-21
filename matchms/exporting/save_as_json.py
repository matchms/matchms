import copy
import json
from typing import List
import numpy
from ..Spectrum import Spectrum


def save_as_json(spectrums: List[Spectrum], filename: str):
    """Save spectrum(s) as json file.

    :py:attr:`~matchms.Spectrum.losses` of spectrum will not be saved.

    Example:

    .. code-block:: python

        import numpy
        from matchms import Spectrum
        from matchms.exporting import save_as_json

        # Create dummy spectrum
        spectrum = Spectrum(mz=numpy.array([100, 200, 300], dtype="float"),
                            intensities=numpy.array([10, 10, 500], dtype="float"),
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
    with open(filename, 'w', encoding="utf-8") as fout:
        json.dump(spectrums, fout, cls=SpectrumJSONEncoder)


def _convert_spectrum_into_dict(spectrum: Spectrum):
    """Convert matchms.Spectrum() into dictionaries"""
    peaks_list = numpy.vstack((spectrum.peaks.mz, spectrum.peaks.intensities)).T.tolist()
    spectrum_dict = {key: spectrum.metadata[key] for key in spectrum.metadata}
    spectrum_dict["peaks_json"] = peaks_list
    return spectrum_dict


class SpectrumJSONEncoder(json.JSONEncoder):
    # See https://github.com/PyCQA/pylint/issues/414 for reference
    def default(self, o):
        """JSON Encoder which can encode a :py:class:`~matchms.Spectrum.Spectrum` object"""
        if isinstance(o, Spectrum):
            spec = o.clone()
            return _convert_spectrum_into_dict(spec)
        return json.JSONEncoder.default(self, o)


class ScoresJSONEncoder(json.JSONEncoder):
    def default(self, o):
        """JSON Encoder which can encode a :py:class:`~matchms.Scores.Scores` object"""
        class_name = o.__class__.__name__
        # do isinstance(o, Scores) without importing matchms.Scores
        if class_name == "Scores":
            scores = copy.deepcopy(o)

            scores_dict = {"__Scores__": True,
                           "similarity_function": str(scores.similarity_function.__class__.__name__),
                           "is_symmetric": scores.is_symmetric,
                           "references": [_convert_spectrum_into_dict(reference) for reference in scores.references],
                           "queries": [_convert_spectrum_into_dict(query) for query in scores.queries] if scores.is_symmetric else None,
                           "scores": scores.scores.tolist()}

            return scores_dict
        return json.JSONEncoder.default(self, o)
