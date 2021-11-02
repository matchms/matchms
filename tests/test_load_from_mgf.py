import os
from io import BytesIO

from matchms.importing import load_from_mgf


def test_load_from_mgf():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "tests", "pesticides.mgf")

    spectra = load_from_mgf(spectra_file)

    assert len(list(spectra)) > 0


def test_load_from_mgf_file_like_obj():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "tests", "pesticides.mgf")

    with open(spectra_file, 'rb') as spectra_bytes:
        buf = BytesIO(spectra_bytes.read())
        spectra = load_from_mgf(spectra_bytes)

        assert len(list(spectra)) > 0
