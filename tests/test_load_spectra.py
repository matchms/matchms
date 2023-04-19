import os
import pytest
from matchms import Spectrum
from matchms.importing.load_spectra import load_spectra


def test_load_spectra_unknown_file(tmp_path):
    """Tests if unknown file raises an Assertion error"""
    with pytest.raises(AssertionError):
        load_spectra(os.path.join(tmp_path, "file_that_does_not_exist.json"))


@pytest.mark.parametrize("filename, ftype, expected_num_spectra", [
    ["pesticides.mgf", None, 76],
    ["testdata.mgf", "mgf", 30],
    ["testdata.mzml", None, 10],
    ["testdata.mzXML", None, 1],
    ["massbank_five_spectra.msp", None, 5]
])
def test_load_spectra(filename, ftype, expected_num_spectra):
    """Test if pickled file is loaded in correctly"""
    tests_root = os.path.join(os.path.dirname(__file__), "../tests")
    file = os.path.join(tests_root, filename)
    actual = list(load_spectra(file, ftype=ftype))
   
        
    assert isinstance(actual, list), "expected list of spectra"
    assert len(actual) == expected_num_spectra
    for spectrum in actual:
        assert isinstance(spectrum, Spectrum)
