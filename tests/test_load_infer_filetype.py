import os
import pytest
from matchms import Spectrum
from matchms.importing.load_infer_filetype import load_spectra


def test_load_spectra_unknown_file(tmp_path):
    """Tests if unknown file raises an Assertion error"""
    with pytest.raises(AssertionError):
        load_spectra(os.path.join(tmp_path, "file_that_does_not_exist.json"))


def test_load_spectra():
    """Test if pickled file is loaded in correctly"""

    tests_root = os.path.join(os.path.dirname(__file__), "../tests")
    spectra_files = [os.path.join(tests_root, "pesticides.mgf"),
                     os.path.join(tests_root, "testdata.mgf"),
                     os.path.join(tests_root, "testdata.mzml"),
                     os.path.join(tests_root, "testdata.mzXML"),
                     os.path.join(tests_root, "massbank_five_spectra.msp"),
                     ]
    for spectrum_file in spectra_files:
        spectra = load_spectra(spectrum_file)
        assert isinstance(spectra, list), "expected list of spectra"
        assert len(spectra) > 0
        for spectrum in spectra:
            assert isinstance(spectrum, Spectrum)
