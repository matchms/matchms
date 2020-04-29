import numpy as np
from matchms import Spectrum
from matchms.filtering import clean_inchis


def test_clean_inchis_misplaced_smiles():
    """Test if misplaced smiles are corrected.
    """
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"inchi": "C1CCCCC1"})

    spectrum = clean_inchis(spectrum_in)
    assert spectrum.get("inchi") == '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"', "Expected different InChI"


def test_clean_inchis_misplaced_inchikey():
    """Test if misplaced inchikeys are corrected.
    """
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"inchi": "InChI=XYLJNLCSTIOKRM-UHFFFAOYSA-N"})

    spectrum = clean_inchis(spectrum_in)
    assert spectrum.get("inchi") == "N/A", "Expected empty InChI"
    assert spectrum.get("inchikey") == "XYLJNLCSTIOKRM-UHFFFAOYSA-N"


def test_clean_inchis_harmonize_strings():
    """Test if inchi strings are made consistent in style.
    """
    spectrum_in1 = Spectrum(mz=np.array([], dtype='float'),
                            intensities=np.array([], dtype='float'),
                            metadata={"inchi": 'InChI=1S/C6H12'})

    spectrum_in2 = Spectrum(mz=np.array([], dtype='float'),
                            intensities=np.array([], dtype='float'),
                            metadata={"inchi": '1S/C6H12'})

    spectrum1 = clean_inchis(spectrum_in1)
    spectrum2 = clean_inchis(spectrum_in2)
    assert spectrum1.get("inchi").startswith('"InChI='), "InChI style not as expected"
    assert spectrum1 == spectrum2, 'after cleaning both spectra should be equal'
