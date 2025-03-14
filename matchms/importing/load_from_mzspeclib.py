from typing import Generator, List, Tuple

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
    'MS:1000045|collision energy': 'collision_energy',
    'MS:1000465|scan polarity': 'ion_mode',
    'MS:1000868|SMILES formula': 'smiles'
}

COMPOSITE_VALUE_MAPPINGS = {
    'UO:0000000|unit': {
        'UO:0000266|electronvolt': 'eV',
        'UO:0000169|parts per million': 'ppm',
        'MS:1000040|m/z': 'mz'
    },
}

LIBRARY_SECTION_PATTERNS = [
    re.compile(r'<mzSpecLib>'),
    re.compile(r'<AttributeSet Spectrum=(.+)>'),
    re.compile(r'<AttributeSet Analyte=(.+)>'),
    re.compile(r'<AttributeSet Interpretation=(.+)>'),
    re.compile(r'<Cluster=(\d+)>'),
    re.compile(r'<Spectrum=(\d+)>'),
]

SPECTRUM_SECTION_PATTERNS = [
    re.compile(r'<Analyte=(\d+)>'),
    re.compile(r'<Interpretation=(\d+)>'),
    re.compile(r'<Peaks>'),
]

INTERPRETATION_SECTION_PATTERNS = [
    re.compile(r'<InterpretationMember=(\d+)>'),
]

ATTRIBUTE_PATTERNS = [
    re.compile(r'(MS:\d+\|[^=]+)=(.+)'),
    re.compile(r'\[(\d+)\](..:\d+\|[^=]+)=(.+)'),
]

def _parse_simple_attribute(line:str) -> Tuple[str, str]:
    matches = ATTRIBUTE_PATTERNS[0].search(line)
    return SPECTRUM_ATTRIBUTE_MAPPING[matches.group(1)], _sanitize(matches.group(2))

def _parse_composite_attribute(lines: List[str]) -> Tuple[str, str]:
    ids = []
    terms = []
    values = []
    for line in lines:
        matches = ATTRIBUTE_PATTERNS[1].search(line)
        ids.append(matches.group(1))
        terms.append(matches.group(2))
        values.append(matches.group(3))
    term = terms[0]
    value = values[0] + COMPOSITE_VALUE_MAPPINGS[terms[1]][values[1]]
    return SPECTRUM_ATTRIBUTE_MAPPING[term], _sanitize(value)


def load_from_mzspeclib(filename: str) -> Generator[Spectrum, None, None]:
    with open(filename, 'r', encoding='UTF-8') as file:
        while line := file.readline():
            if LIBRARY_SECTION_PATTERNS[5].search(line):
                spectrum_attributes = {}
                while line := file.readline():
                    if ATTRIBUTE_PATTERNS[1].search(line):
                        continue
                    elif ATTRIBUTE_PATTERNS[0].search(line):
                        key, val = _parse_simple_attribute(line)
                        spectrum_attributes[key] = val
                        continue                       
                    if '<Peaks>' in line:
                        num_peaks = int(spectrum_attributes['num peaks'])
                        lines = [file.readline() for i in range(num_peaks)]
                        mzs, intensities, comments = _read_peaks(lines)
                        spectrum_attributes['peak_comments'] = comments
                        yield Spectrum(mzs, intensities, spectrum_attributes)
                        break

def _read_peaks(lines: List[str]):
    mzs = []
    intensities = []
    comments = {}
    for x in lines:
        mz, intensity, comment = x.split('\t')

        mzs.append(float(mz))
        intensities.append(float(intensity))
        comments[mz] = comment
    mz_array = np.array(mzs, dtype=np.float32)
    intensity_array = np.array(intensities, dtype=np.float32)
    return mz_array, intensity_array, comments

def _sanitize(value: str) -> str:
    return value.strip().strip('\n').strip('\t')