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
    with open(filename, 'w') as fout:
        json.dump(spectrums, fout, cls=SpectrumJSONEncoder)


class SpectrumJSONEncoder(json.JSONEncoder):
    # pylint: disable=method-hidden
    # See https://github.com/PyCQA/pylint/issues/414 for reference
    def default(self, o):
        """JSON Encoder which can encode a :py:class:`~matchms.Spectrum.Spectrum` object"""
        if isinstance(o, Spectrum):
            spec = o.clone()
            peaks_list = numpy.vstack((spec.peaks.mz, spec.peaks.intensities)).T.tolist()

            # Convert matchms.Spectrum() into dictionaries
            spectrum_dict = {key: spec.metadata[key] for key in spec.metadata}
            spectrum_dict["peaks_json"] = peaks_list
            return spectrum_dict
        return json.JSONEncoder.default(self, o)
