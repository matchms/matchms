import numpy as np
from matchms import Spectrum
from matchms.filtering import clean_inchikeys


def test_clean_inchikeys_empty_harmonize():
    """Test if empty entry is harmonized.
    """
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"inchikey": "no data"})

    spectrum = clean_inchikeys(spectrum_in)
    assert spectrum.get("inchikey") == "N/A", "Expected empty entry"


def test_clean_inchikeys_defect_entry_harmonize():
    """Test if empty entry is harmonized.
    """
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"inchikey": "77LJNLCSTIOKRM-UHFFFAOYSA-N"})

    spectrum = clean_inchikeys(spectrum_in)
    assert spectrum.get("inchikey") == "N/A", "Expected empty entry"


def test_clean_inchikeys_misplaced_inchiaux():
    """Test if misplaced smiles are corrected.
    """
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"inchiaux": "XYLJNLCSTIOKRM-UHFFFAOYSA-N"})

    spectrum = clean_inchikeys(spectrum_in)
    assert spectrum.get("inchikey") == "XYLJNLCSTIOKRM-UHFFFAOYSA-N", "Expected different inchikey"


def test_clean_inchikeys_defect_inchiaux():
    """Test if misplaced smiles are corrected.
    """
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"inchikey": "none",
                                     "inchiaux": "XYLJNLCSTIOKRM-UHFFFAOYSA-XX"})

    spectrum = clean_inchikeys(spectrum_in)
    assert spectrum.get("inchikey") == "N/A", "Expected empty entry"
