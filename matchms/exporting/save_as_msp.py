from typing import List
from ..Spectrum import Spectrum


def save_as_msp(spectra: List[Spectrum], filename: str):
    """Save spectrum(s) as msp file.

    :py:attr:`~matchms.Spectrum.losses` of spectrum will not be saved.

    Example:

    .. code-block:: python

        import numpy
        from matchms import Spectrum
        from matchms.exporting import save_as_msp

        # Create dummy spectrum
        spectrum = Spectrum(mz=numpy.array([100, 200, 300], dtype="float"),
                            intensities=numpy.array([10, 10, 500], dtype="float"),
                            metadata={"charge": -1,
                                      "inchi": '"InChI=1S/C6H12"',
                                      "precursor_mz": 222.2})

        # Write spectrum to test file
        save_as_msp(spectrum, "test.msp")

    Parameters
    ----------
    spectra:
        Expected input are match.Spectrum.Spectrum() objects.
    filename:
        Provide filename to save spectrum(s).
    """

    assert filename.endswith('.msp'), "File extension must be 'msp'."

    spectra = ensure_list(spectra)

    with open(filename, 'w') as outfile:
        for spectrum in spectra:
            for key, value in spectrum.metadata:
                outfile.write('%s:%s\n' % (key, value))


def ensure_list(spectra) -> List[Spectrum]:
    if not isinstance(spectra, list):
        # Assume that input was single Spectrum
        spectra = [spectra]
    return spectra
