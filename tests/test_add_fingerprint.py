import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import add_fingerprint
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize('metadata, expected', [
    [{"smiles": "[C+]#C[O-]"}, numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])],
    [{"inchi": "InChI=1S/C2O/c1-2-3"}, numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])],
    [{}, None]
])
def test_add_fingerprint(metadata, expected):
    pytest.importorskip("rdkit")
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = add_fingerprint(spectrum_in, nbits=16)
    assert numpy.all(spectrum.get("fingerprint") == expected), "Expected different fingerprint."


def test_add_fingerprint_empty_spectrum():
    """Test if empty spectrum is handled correctly."""

    spectrum = add_fingerprint(None)
    assert spectrum is None, "Expected None."
