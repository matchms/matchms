from typing import List
from matchms.Spectrum import Spectrum
import os

def save_as_mzspeclib(spectra: List[Spectrum], filename: str):

    with open(filename, 'w') as file:
        _write_header(filename, file)
        for idx, spectrum in enumerate(spectra):
            _write_spectrum(file, idx, spectrum)


def _write_spectrum(file, idx, spectrum):
    print(f'<Spectrum={idx + 1}>', file=file)
    print(f'MS:1003059|number of peaks={len(spectrum.peaks)}', file=file)
    print(f'<Peaks>', file=file)
    for i in range(len(spectrum.peaks)):
        print(f'{spectrum.peaks.mz[i]}\t{spectrum.peaks.intensities[i]}\t?', file=file)


def _write_header(filename, file):
    basename, ext = os.path.splitext(filename)
    name = basename.split(os.path.sep)[-1]
    print('<mzSpecLib>', file=file)
    print('MS:1003186|library format version=1.0', file=file)
    print(f'MS:1003188|library name={name}', file=file)
    print('<AttributeSet Spectrum=all>', file=file)
    print('<AttributeSet Analyte=all>', file=file)
    print('<AttributeSet Interpretation=all>', file=file)