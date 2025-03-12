import os
import re
from typing import List, TextIO
from matchms.Spectrum import Spectrum


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

MAPPED_SPECTRUM_ATTRIBUTES = {
    'MS:1000465|scan polarity': {
        'positive': 'MS:1000130|positive scan',
        'negative': 'MS:1000129|negative scan'
    }
}

STANDARDIZED_SPECTRUM_ATTRIBUTES = {
    'collision_energy': ['MS:1000045|collision energy', 'UO:0000266|electronvolt'],
}


def save_as_mzspeclib(spectra: List[Spectrum], filename: str) -> None:
    """
    Save a list of spectra to a file in mzSpecLib format.

    Parameters:
    spectra (List[Spectrum]): List of Spectrum objects to save.
    filename (str): The name of the file to save the spectra to.
    """
    with open(filename, 'w', encoding='UTF-8') as file:
        _write_header(filename, file)
        for idx, spectrum in enumerate(spectra):
            _write_spectrum(file, idx, spectrum)

def _write_spectrum(file: TextIO, idx: int, spectrum: Spectrum) -> None:
    """
    Write a single spectrum to the file.

    Parameters:
    file (TextIO): The file object to write to.
    idx (int): The index of the spectrum in the list.
    spectrum (Spectrum): The Spectrum object to write.
    """
    print(f'<Spectrum={idx + 1}>', file=file)
    _write_spectrum_attributes(file, spectrum)
    if _has_analyte(spectrum):
        _write_analyte(file, spectrum)
    _write_peaks(file, spectrum)
    print('', file=file)

def _write_analyte(file: TextIO, spectrum: Spectrum) -> None:
    """
    Write analyte information for a spectrum to the file.

    Parameters:
    file (TextIO): The file object to write to.
    spectrum (Spectrum): The Spectrum object containing analyte information.
    """
    print('<Analyte=1>', file=file)
    for key, attribute in ANALYTE_ATTRIBUTES.items():
        value = spectrum.get(key)
        if value is not None:
            print(f'{attribute}={value}', file=file)

def _write_spectrum_attributes(file: TextIO, spectrum: Spectrum) -> None:
    """
    Write spectrum attributes to the file.

    Parameters:
    file (TextIO): The file object to write to.
    spectrum (Spectrum): The Spectrum object containing attributes.
    """
    _write_defined_spectrum_attributes(file, spectrum)
    
    spectrum_attributes = spectrum.metadata.keys()
    attr_counter = 1
    
    for attr in spectrum_attributes:
        if attr in STANDARDIZED_SPECTRUM_ATTRIBUTES:
            _write_spectrum_attribute_with_unit(file, spectrum, attr_counter, attr)
            attr_counter += 1
        elif attr not in SPECTRUM_ATTRIBUTES and attr not in ANALYTE_ATTRIBUTES:
            _write_other_spectrum_attribute(file, spectrum, attr_counter, attr)
            attr_counter += 1

    print(f'MS:1003059|number of peaks={len(spectrum.peaks)}', file=file)

def _write_other_spectrum_attribute(file: TextIO, spectrum: Spectrum, attr_counter: int, attr: str) -> None:
    """
    Write other spectrum attributes to the file.

    Parameters:
    file (TextIO): The file object to write to.
    spectrum (Spectrum): The Spectrum object containing attributes.
    attr_counter (int): The counter for the attribute.
    attr (str): The attribute name.
    """
    value = spectrum.get(attr)
    print(f'[{attr_counter}]MS:1003275|other attribute name={attr}', file=file)
    print(f'[{attr_counter}]MS:1003276|other attribute value={value}', file=file)

def _write_spectrum_attribute_with_unit(file: TextIO, spectrum: Spectrum, attr_counter: int, attr: str) -> None:
    """
    Write spectrum attributes with units to the file.

    Parameters:
    file (TextIO): The file object to write to.
    spectrum (Spectrum): The Spectrum object containing attributes.
    attr_counter (int): The counter for the attribute.
    attr (str): The attribute name.
    """
    term, unit = STANDARDIZED_SPECTRUM_ATTRIBUTES.get(attr)
    # remove non-numeric unit identifiers from value
    value = _extract_numeric_value(spectrum.get(attr)) 
    print(f'[{attr_counter}]{term}={value}', file=file)
    print(f'[{attr_counter}]UO:0000000|unit={unit}', file=file)

def _extract_numeric_value(value: str) -> str:
    """
    Extract numeric value from a string.

    Parameters:
    value (str): The string containing numeric value.

    Returns:
    str: The extracted numeric value.
    """
    value = re.findall('[\\d]+[.,\\d]+|[\\d]*[.][\\d]+|[\\d]+', value)[0]
    return value

def _write_defined_spectrum_attributes(file: TextIO, spectrum: Spectrum) -> None:
    """
    Write defined spectrum attributes to the file.

    Parameters:
    file (TextIO): The file object to write to.
    spectrum (Spectrum): The Spectrum object containing attributes.
    """
    for key, attribute in SPECTRUM_ATTRIBUTES.items():
        value = spectrum.get(key)
        if attribute in MAPPED_SPECTRUM_ATTRIBUTES:
            value = MAPPED_SPECTRUM_ATTRIBUTES[attribute].get(value)        
        if value is not None:
            print(f'{attribute}={value}', file=file)

def _write_peaks(file: TextIO, spectrum: Spectrum) -> None:
    """
    Write peaks information for a spectrum to the file.

    Parameters:
    file (TextIO): The file object to write to.
    spectrum (Spectrum): The Spectrum object containing peaks information.
    """
    print('<Peaks>', file=file)
    peak_comments = spectrum.get('peak_comments', {})
    for i in range(len(spectrum.peaks)):
        mz = spectrum.peaks.mz[i]
        intensities = f'{spectrum.peaks.intensities[i]:.2f}'.rstrip('0').rstrip('.')
        comment = peak_comments.get(mz, '?')
        print(f'{mz}\t{intensities}\t{comment}', file=file)

def _has_analyte(spectrum: Spectrum) -> bool:
    """
    Check if a spectrum has analyte information.

    Parameters:
    spectrum (Spectrum): The Spectrum object to check.

    Returns:
    bool: True if the spectrum has analyte information, False otherwise.
    """
    return any(spectrum.get(key) for key in ANALYTE_ATTRIBUTES)

def _write_header(filename: str, file: TextIO) -> None:
    """
    Write the header information to the file.

    Parameters:
    filename (str): The name of the file.
    file (TextIO): The file object to write to.
    """
    basename, _ = os.path.splitext(filename)
    name = basename.split(os.path.sep)[-1]
    print('<mzSpecLib>', file=file)
    print('MS:1003186|library format version=1.0', file=file)
    print(f'MS:1003188|library name={name}', file=file)
    print('<AttributeSet Spectrum=all>', file=file)
    print('<AttributeSet Analyte=all>', file=file)
    print('<AttributeSet Interpretation=all>', file=file)
