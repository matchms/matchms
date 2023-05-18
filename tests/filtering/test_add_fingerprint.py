import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from testfixtures import LogCapture
from matchms.exporting import save_as_json, save_as_mgf, save_as_msp
from matchms.filtering import add_fingerprint
from matchms.importing import load_from_json, load_from_mgf, load_from_msp
from matchms.logging_functions import (reset_matchms_logger,
                                       set_matchms_logger_level)
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize('metadata, expected', [
    [{"smiles": "[C+]#C[O-]"}, np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])],
    [{"inchi": "InChI=1S/C2O/c1-2-3"}, np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])]
])
def test_add_fingerprint(metadata, expected):
    pytest.importorskip("rdkit")
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = add_fingerprint(spectrum_in, nbits=16)
    assert np.all(spectrum.get("fingerprint") == expected), "Expected different fingerprint."


def test_add_fingerprint_no_smiles_no_inchi():
    """Test if fingerprint it generated correctly."""
    set_matchms_logger_level("INFO")
    spectrum_in = SpectrumBuilder().with_metadata({"compound_name": "test name"}).build()

    with LogCapture() as log:
        spectrum = add_fingerprint(spectrum_in)
    assert spectrum.get("fingerprint", None) is None, "Expected None."
    log.check(
        ("matchms", "INFO", "No fingerprint was added (name: test name).")
    )
    reset_matchms_logger()


def test_add_fingerprint_empty_spectrum():
    """Test if empty spectrum is handled correctly."""

    spectrum = add_fingerprint(None)
    assert spectrum is None, "Expected None."


@pytest.mark.parametrize("export_function, expected_filename, load_function", [(save_as_msp, "massbank_five_spectra.msp", load_from_msp),
                                                         (save_as_mgf, "test_remove_fingerprint.mgf", load_from_mgf),
                                                         (save_as_json, "test_remove_fingerprint.json", load_from_json)])
def test_remove_fingerprint_from_metadata(export_function, expected_filename, load_function):
    pytest.importorskip("rdkit")
    module_root = os.path.join(os.path.dirname(__file__), "..")
    expected = list(load_function(os.path.join(module_root, "testdata", expected_filename)))
    spectrum = list(load_from_msp(os.path.join(module_root, "testdata", "massbank_five_spectra.msp")))
    spectrum = list(map(add_fingerprint, spectrum))

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, f"test{Path(expected_filename).suffix}")
        export_function(spectrum, filename)
        actual = list(load_function(filename))
        assert expected == actual