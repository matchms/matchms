from typing import List
from matchms.Spectrum import Spectrum
import os

ANALYTE_ATTRIBUTES = {
    'formula': 'MS:1000866|molecular formula',
    'smiles': 'MS:1000868|SMILES formula',
    'inchi': 'MS:1003403|InChI',
    'inchikey': 'MS:1002894|InChIKey'
}

SPECTRUM_ATTRIBUTES = {
    'compound_name': 'MS:1003061|library spectrum name',
    'precursor_mz': 'MS:1003208|experimental precursor monoisotopic m/z',
    'scans': 'MS:1003057|scan number',
    'charge': 'MS:1000041|charge state',
    'ionmode': 'MS:1000465|scan polarity'
}

MAPPED_ATTRIBUTE_VALUES = {
    'MS:1000465|scan polarity': {
        'positive': 'MS:1000130|positive scan',
        'negative': 'MS:1000129|negative scan'
    }
}


def save_as_mzspeclib(spectra: List[Spectrum], filename: str):
    with open(filename, 'w') as file:
        _write_header(filename, file)
        for idx, spectrum in enumerate(spectra):
            _write_spectrum(file, idx, spectrum)

def _write_spectrum(file, idx, spectrum):
    print(f'<Spectrum={idx + 1}>', file=file)
    _write_spectrum_attributes(file, spectrum)
    _write_analyte(file, spectrum)
    _write_peaks(file, spectrum)
    print('', file=file)

def _write_analyte(file, spectrum):
    if _has_analyte(spectrum):
        print('<Analyte=1>', file=file)
        for key, attribute in ANALYTE_ATTRIBUTES.items():
            value = spectrum.get(key)
            if value is not None:
                print(f'{attribute}={value}', file=file)

def _write_spectrum_attributes(file, spectrum):
    for key, attribute in SPECTRUM_ATTRIBUTES.items():
        value = spectrum.get(key)
        if attribute in MAPPED_ATTRIBUTE_VALUES.keys():
            value = MAPPED_ATTRIBUTE_VALUES[attribute].get(value)
        if value is not None:
            print(f'{attribute}={value}', file=file)
    
    spectrum_attributes = spectrum.metadata.keys()
    attr_counter = 1
    for attr in spectrum_attributes:
        if attr not in SPECTRUM_ATTRIBUTES and attr not in ANALYTE_ATTRIBUTES:
            value = spectrum.get(attr)
            print(f'[{attr_counter}]MS:1003275|other attribute name={attr}', file=file)
            print(f'[{attr_counter}]MS:1003276|other attribute value={value}', file=file)
            attr_counter += 1

    print(f'MS:1003059|number of peaks={len(spectrum.peaks)}', file=file)

def _write_peaks(file, spectrum):
    print(f'<Peaks>', file=file)
    for i in range(len(spectrum.peaks)):
        intensities = '{0:.2f}'.format(spectrum.peaks.intensities[i]).rstrip('0').rstrip('.')
        print(f'{spectrum.peaks.mz[i]}\t{intensities}\t?', file=file)

def _has_analyte(spectrum):
    return any([spectrum.get(key) for key in ANALYTE_ATTRIBUTES.keys()])

def _write_header(filename, file):
    basename, ext = os.path.splitext(filename)
    name = basename.split(os.path.sep)[-1]
    print('<mzSpecLib>', file=file)
    print('MS:1003186|library format version=1.0', file=file)
    print(f'MS:1003188|library name={name}', file=file)
    print('<AttributeSet Spectrum=all>', file=file)
    print('<AttributeSet Analyte=all>', file=file)
    print('<AttributeSet Interpretation=all>', file=file)
