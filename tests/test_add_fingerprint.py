import numpy as np
import pytest
from testfixtures import LogCapture
from matchms.filtering import add_fingerprint
from matchms.logging_functions import (reset_matchms_logger,
                                       set_matchms_logger_level)
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize('metadata, expected', [
    [{"smiles": "[C+]#C[O-]"}, numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])],
    [{"inchi": "InChI=1S/C2O/c1-2-3"}, numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])]
])
def test_add_fingerprint(metadata, expected):
    pytest.importorskip("rdkit")
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = add_fingerprint(spectrum_in, nbits=16)
    assert numpy.all(spectrum.get("fingerprint") == expected), "Expected different fingerprint."


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
