import os
import numpy
import pytest

from matchms import Spectrum
from matchms.importing import load_from_msp
from matchms.exporting import save_as_msp


@pytest.fixture
def none_spectrum():
    return None


@pytest.fixture
def spectrum():
    return Spectrum(mz=numpy.array([100, 200, 290, 490, 510], dtype="float"),
                    intensities=numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"))


@pytest.fixture
def empty_spectrum():
    return Spectrum(mz=numpy.array([], dtype="float"),
                    intensities=numpy.array([], dtype="float"))


@pytest.fixture
def rcx_gc_spectra():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "tests", "rcx_gc-ei_ms_20201028_perylene.msp")
    spectra = load_from_msp(spectrums_file)
    return spectra


def test_spectrum_none_exception(none_spectrum, tmp_path):
    """ Test for exception being thrown if the spectrum to be saved. """
    filename = os.path.join(tmp_path, "test.msp")

    with pytest.raises(AttributeError) as exception:
        save_as_msp(none_spectrum, filename)

    message = exception.value.args[0]
    assert message == "'NoneType' object has no attribute 'metadata'"


def test_wrong_filename_exception(tmp_path):
    """ Test for exception being thrown if output file doesn't end with .msp. """
    filename = os.path.join(tmp_path, "test.mzml")

    with pytest.raises(AssertionError) as exception:
        save_as_msp(None, filename)

    message = exception.value.args[0]
    assert message == "File extension must be 'msp'."


# Using tmp_path fixture from pytest: https://docs.pytest.org/en/stable/tmpdir.html#the-tmp-path-fixture
def test_file_exists_single_spectrum(spectrum, tmp_path):
    filename = os.path.join(tmp_path, "test.msp")
    save_as_msp(spectrum, filename)
    assert os.path.isfile(filename)
