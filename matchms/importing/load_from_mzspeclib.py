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
}

COMPOSITE_VALUE_MAPPINGS = {
    'UO:0000000|unit': {'UO:0000266|electronvolt': 'eV'},
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
    ids, terms, values = [ATTRIBUTE_PATTERNS[1].search(x) for x in lines]


def load_from_mzspeclib(filename: str) -> Generator[Spectrum, None, None]:
    spectrum_id = None

    with open(filename, 'r', encoding='UTF-8') as file:
        for line in file:
            match = re.compile(r'<Spectrum=(\d+)>').search(line)
            if match:
                spectrum_attributes = {}
                line = file.readline()

            # number_match = re.compile(r'\[(\d+)\]').search(line)
            # if number_match:
            #     accessions = []
            #     values = []
            #     attr_number = int(number_match.group(1))
            #     current_attr_number, accession, value = _parse_composite_attribute_line(line)
            #     accessions.append(accession)
            #     values.append(value)
            #     continue

            attr_match = re.compile(r'(MS:\d+\|[^=]+)=(.+)').search(line)
            if attr_match:
                tokens = line.split('=')
                if len(tokens) == 2:
                    term, value = tokens
                    if term in SPECTRUM_ATTRIBUTE_MAPPING:
                        spectrum_attributes[SPECTRUM_ATTRIBUTE_MAPPING[term]] = _sanitize(value)
                        continue
            
            analyte_match = re.compile(r'<Analyte=(\d+)>').search(line)
            if analyte_match:
                accession = line.split('|')[0]
                if accession == 'MS:1000868':
                    spectrum_attributes['smiles'] = line.strip('MS:1000868|SMILES formula=')
                    continue
                
            if '<Peaks>' in line:
                num_peaks = int(spectrum_attributes['num peaks'])
                mzs, intensities, comments = _read_peaks(file, num_peaks)
                spectrum_attributes['peak_comments'] = comments
                yield Spectrum(mzs, intensities, spectrum_attributes)

def _read_peaks(file, num_peaks):
    mzs = []
    intensities = []
    comments = {}
    for i in range(num_peaks):
        line = file.readline()
        mz, intensity, comment = line.split('\t')

        mzs.append(float(mz))
        intensities.append(float(intensity))
        comments[mz] = comment
    mz_array = np.array(mzs, dtype=np.float32)
    intensity_array = np.array(intensities, dtype=np.float32)
    return mz_array, intensity_array, comments

def _sanitize(value: str) -> str:
    return value.strip().strip('\n').strip('\t')

def _parse_composite_attribute_line(line: str):
    pattern = re.compile(r'\[(\d+)\](..:\d+\|[^=]+)=(.+)')
    match = pattern.match(line)
    if match:
        attribute_number = int(match.group(1))
        accession_term = match.group(2)
        value = match.group(3)
        return attribute_number, accession_term, value
    return None