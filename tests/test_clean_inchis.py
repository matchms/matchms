import numpy as np
from matchms import Spectrum
from matchms.filtering import clean_inchis


def test_clean_inchis_misplaced_smiles():
    """Test if misplaced smiles are corrected.
    """
    spectrum_in = Spectrum(mz=np.array([10, 20, 30, 40], dtype='float'),
                           intensities=np.array([0, 1, 10, 100], dtype='float'),
                           metadata={"inchi": "C1CCCCC1"})

    spectrum = clean_inchis(spectrum_in)
    assert spectrum.get("inchi") == '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"', "Expected different InChI"


def test_clean_inchis_harmonize_strings():
    """Test if inchi strings are made consistent in style.
    """
    spectrum_in1 = Spectrum(mz=np.array([10, 20, 30, 40], dtype='float'),
                            intensities=np.array([0, 1, 10, 100], dtype='float'),
                            metadata={"inchi": 'InChI=1S/C6H12'})

    spectrum_in2 = Spectrum(mz=np.array([10, 20, 30, 40], dtype='float'),
                            intensities=np.array([0, 1, 10, 100], dtype='float'),
                            metadata={"inchi": '1S/C6H12'})

    spectrum1 = clean_inchis(spectrum_in1)
    spectrum2 = clean_inchis(spectrum_in2)
    assert spectrum1.get("inchi").startswith('"InChI='), "InChI style not as expected"
    assert spectrum2.get("inchi").startswith('"InChI='), "InChI style not as expected"


if __name__ == '__main__':
    test_clean_inchis_misplaced_smiles()
    test_clean_inchis_harmonize_strings()
