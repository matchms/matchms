import os
from typing import IO
from typing import List
from ..Spectrum import Spectrum
from ..Spikes import Spikes


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
            write_spectrum(spectrum, outfile)


def write_spectrum(spectrum: Spectrum, outfile: IO):
    write_metadata(spectrum.metadata, outfile)
    write_peaks(spectrum.peaks, outfile)
    outfile.write(os.linesep)


def write_peaks(peaks: Spikes, outfile: IO):
    outfile.write('NUM PEAKS: %d\n' % len(peaks))
    for mz, intensity in zip(peaks.mz, peaks.intensities):
        outfile.write('%s\t%s\n' % (str(mz), str(intensity)))


def write_metadata(metadata: dict, outfile: IO):
    for key, value in metadata.items():
        if not is_num_peaks(key):
            outfile.write('%s: %s\n' % (key.upper(), str(value)))


def is_num_peaks(key: str) -> bool:
    return key.lower().startswith("num peaks")


def ensure_list(spectra) -> List[Spectrum]:
    if not isinstance(spectra, list):
        # Assume that input was single Spectrum
        spectra = [spectra]
    return spectra
