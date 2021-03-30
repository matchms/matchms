import os
import tempfile
import numpy
import pytest
from typing import List

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


@pytest.fixture(params=["rcx_gc-ei_ms_20201028_perylene.msp", "MoNA-export-GC-MS-first10.msp"])
def data(request):
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "tests", request.param)
    spectra = load_from_msp(spectrums_file)
    return list(spectra)


def test_spectrum_none_exception(none_spectrum, tmp_path):
    """ Test for exception being thrown if the spectrum to be saved. """
    filename = tempfile.TemporaryFile(dir=tmp_path, suffix=".msp").name

    with pytest.raises(AttributeError) as exception:
        save_as_msp(none_spectrum, filename)

    message = exception.value.args[0]
    assert message == "'NoneType' object has no attribute 'metadata'"


def test_wrong_filename_exception(tmp_path):
    """ Test for exception being thrown if output file doesn't end with .msp. """
    filename = tempfile.TemporaryFile(dir=tmp_path, suffix=".mzml").name

    with pytest.raises(AssertionError) as exception:
        save_as_msp(None, filename)

    message = exception.value.args[0]
    assert message == "File extension must be 'msp'."


# Using tmp_path fixture from pytest: https://docs.pytest.org/en/stable/tmpdir.html#the-tmp-path-fixture
def test_file_exists_single_spectrum(spectrum, tmp_path):
    """ Test checking if the file is created. """
    filename = tempfile.TemporaryFile(dir=tmp_path, suffix=".msp").name
    save_as_msp(spectrum, filename)
    assert os.path.isfile(filename)


def test_stores_all_spectra(data, tmp_path):
    """ Test checking if all spectra contained in the original file are stored
    and loaded back in properly. """
    _, spectra = save_and_reload_spectra(tmp_path, data)

    assert len(spectra) == len(data)


def test_have_metadata(data, tmp_path):
    """ Test checking of all metadate is stored correctly. """
    _, spectra = save_and_reload_spectra(tmp_path, data)

    assert len(spectra) == len(data)

    for actual, expected in zip(spectra, data):
        assert actual.metadata == expected.metadata


def test_have_peaks(data, tmp_path):
    """ Test checking if all peaks are stored correctly. """
    _, spectra = save_and_reload_spectra(tmp_path, data)

    assert len(spectra) == len(data)

    for actual, expected in zip(spectra, data):
        assert actual.peaks == expected.peaks


def save_and_reload_spectra(tmp_path, spectra: List[Spectrum]):
    """ Utility function to save spectra to msp and load them again.

    Params:
    -------
    tmp_path : Temporary directory where to store the msp file.
    spectra: Spectra objects to store

    Returns:
    --------
    filename: Filename of msp file containing the stored spectra.
    reloaded_spectra: Spectra loaded from saved msp file.
    """

    filename = tempfile.TemporaryFile(dir=tmp_path, suffix=".msp").name
    save_as_msp(spectra, filename)
    reloaded_spectra = list(load_from_msp(filename))
    return filename, reloaded_spectra
