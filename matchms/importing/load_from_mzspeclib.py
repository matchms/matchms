from typing import Generator, List

import numpy as np
from matchms.Spectrum import Spectrum
import re

SPECTRUM_ATTRIBUTE_MAPPING = {
    'MS:1003061|library spectrum name': 'compound_name',
    'MS:1003208|experimental precursor monoisotopic m/z': 'precursor_mz',
    'MS:1003059|number of peaks': 'num peaks',
    'MS:1000866|molecular formula': 'formula',
    'MS:1002894|InChIKey': 'inchikey',
    'MS:1003403|InChI': 'inchi',
}

def load_from_mzspeclib(filename: str) -> Generator[Spectrum, None, None]:
    spectrum_id = None

    with open(filename, 'r', encoding='UTF-8') as file:
        for line in file:
            match = re.compile(r'<Spectrum=(\d+)>').search(line)
            if match:
                if spectrum_id is not None:
                    yield Spectrum(
                        np.array(mzs, dtype=np.float32),
                        np.array(intensities, dtype=np.float32),
                        params)
                spectrum_id = int(match.group(1))
                mzs = []
                intensities = []
                comments = []
                params = {}
                continue

            if spectrum_id is not None:
                if '<Peaks>' in line:
                    num_peaks = int(params['num peaks'])
                    for i in range(num_peaks):
                        line = file.readline()
                        mz, intensity, comment = line.split('\t')

                        mzs.append(float(mz))
                        intensities.append(float(intensity))
                        comments.append(comment)

                accession = line.split('|')[0]
                if accession == 'MS:1000868':
                    continue
                tokens = line.split('=')
                if len(tokens) == 2:
                    term, value = tokens
                    if term in SPECTRUM_ATTRIBUTE_MAPPING:
                        params[SPECTRUM_ATTRIBUTE_MAPPING[term]] = _sanitize(value)
    yield Spectrum(
        np.array(mzs, dtype=np.float32),
        np.array(intensities, dtype=np.float32),
        params)

def _sanitize(value:str) -> str:
    return value.strip().strip('\n').strip('\t')