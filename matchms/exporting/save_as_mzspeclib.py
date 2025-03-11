from typing import List
from matchms.Spectrum import Spectrum
import os

def save_as_mzspeclib(spectra: List[Spectrum], filename: str):
    basename, ext = os.path.splitext(filename)
    name = basename.split(os.path.sep)[-1]
    with open(filename, 'w') as file:
        print('<mzSpecLib>', file=file)
        print('MS:1003186|library format version=1.0', file=file)
        print(f'MS:1003188|library name={name}', file=file)