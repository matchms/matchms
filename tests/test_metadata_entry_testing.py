import numpy as np
from matchms import Spectrum
from matchms.metadata_entry_testing import entry_is_empty, is_valid_inchikey


def test_entry_is_empty_inchi_key():
    """Test if empty inchi and inchikey fields are recognized."""
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"inchi": '"InChI="'})

    assert entry_is_empty(spectrum_in, "inchi"), "Expected empty entry"
    assert entry_is_empty(spectrum_in, "inchikey"), "Expected empty entry"


def test_entry_is_empty_smiles():
    """Test if empty smiles field is recognized."""
    spectrum_in = Spectrum(mz=np.array([], dtype='float'),
                           intensities=np.array([], dtype='float'),
                           metadata={"smiles": "n/a"})
    assert entry_is_empty(spectrum_in, "smiles"), "Expected empty entry"


def test_is_valid_inchikey():
    """Test if strings are correctly classified."""
    inchikeys_true = ["XYLJNLCSTIOKRM-UHFFFAOYSA-N",
                      "XYLJNLCSTIOKRM-aaaaaaaaaa-a"]
    inchikeys_false = ["XYLJNLCSTIOKRM-UHFFFAOYSA",
                       "XYLJNLCSTIOKRMRUHFFFAOYSASN",
                       "XYLJNLCSTIOKR-MUHFFFAOYSA-N",
                       "XYLJNLCSTIOKRM-UHFFFAOYSA-NN",
                       "Brcc(NC2=NCN2)-ccc3nccnc1-3",
                       "2YLJNLCSTIOKRM-UHFFFAOYSA-N"]

    for inchikey in inchikeys_true:
        assert is_valid_inchikey(inchikey), "Expected inchikey is True."
    for inchikey in inchikeys_false:
        assert not is_valid_inchikey(inchikey), "Expected inchikey is False."
