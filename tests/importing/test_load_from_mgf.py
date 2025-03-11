import os
from pathlib import Path
import pytest
from matchms import Spectrum
from matchms.importing import load_from_mgf


def test_load_from_mgf_using_filepath():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", "pesticides.mgf")

    spectra = list(load_from_mgf(spectra_file))

    assert len(spectra) > 0
    assert isinstance(spectra[0], Spectrum)

    spectra_file_path = Path(module_root).joinpath("testdata", "pesticides.mgf")
    spectra = list(load_from_mgf(spectra_file_path))

    assert len(spectra) > 0
    assert isinstance(spectra[0], Spectrum)


def test_load_missing_mgf_raises():
    with pytest.raises(FileNotFoundError):
        load_from_mgf('does-not-exist.mgf')
