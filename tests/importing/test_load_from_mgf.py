import os
from matchms import Spectrum
from matchms.importing import load_from_mgf


def test_load_from_mgf_using_filepath():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", "pesticides.mgf")

    spectra = list(load_from_mgf(spectra_file))

    assert len(spectra) > 0
    assert isinstance(spectra[0], Spectrum)


def test_load_from_mgf_using_file():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_filepath = os.path.join(module_root, "testdata", "pesticides.mgf")

    with open(spectra_filepath, "r", encoding="utf-8") as spectra_file:
        spectra = list(load_from_mgf(spectra_file))

        assert len(spectra) > 0
        assert isinstance(spectra[0], Spectrum)
