import os
import tempfile
from typing import List
import numpy
import pytest
from matchms import Spectrum
from matchms.exporting import save_as_msp
from matchms.importing import load_from_msp


@pytest.fixture
def none_spectrum():
    return None


@pytest.fixture
def spectrum():
    return Spectrum(mz=numpy.array([100, 200, 290, 490, 510], dtype="float"),
                    intensities=numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"))


@pytest.fixture(params=["rcx_gc-ei_ms_20201028_perylene.msp", "MoNA-export-GC-MS-first10.msp", "Hydrogen_chloride.msp"])
def data(request):
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "tests", request.param)
    spectra = load_from_msp(spectrums_file)
    return list(spectra)


@pytest.yield_fixture
def filename():
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "test.msp")
        yield filename


def test_spectrum_none_exception(none_spectrum, filename):
    """ Test for exception being thrown if the spectrum to be saved. """
    with pytest.raises(AttributeError) as exception:
        save_as_msp(none_spectrum, filename)

    message = exception.value.args[0]
    assert message == "'NoneType' object has no attribute 'metadata'"


def test_wrong_filename_exception():
    """ Test for exception being thrown if output file doesn't end with .msp. """
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "test.mzml")

        with pytest.raises(AssertionError) as exception:
            save_as_msp(None, filename)

        message = exception.value.args[0]
        assert message == "File extension must be 'msp'."


# Using tmp_path fixture from pytest: https://docs.pytest.org/en/stable/tmpdir.html#the-tmp-path-fixture
def test_file_exists_single_spectrum(spectrum, filename):
    """ Test checking if the file is created. """
    save_as_msp(spectrum, filename)

    assert os.path.isfile(filename)


def test_stores_all_spectra(filename, data):
    """ Test checking if all spectra contained in the original file are stored
    and loaded back in properly. """
    spectra = save_and_reload_spectra(filename, data)

    assert len(spectra) == len(data)


def test_have_metadata(filename, data):
    """ Test checking of all metadate is stored correctly. """
    spectra = save_and_reload_spectra(filename, data)

    assert len(spectra) == len(data)

    for actual, expected in zip(spectra, data):
        assert actual.metadata == expected.metadata


def test_have_peaks(filename, data):
    """ Test checking if all peaks are stored correctly. """
    spectra = save_and_reload_spectra(filename, data)

    assert len(spectra) == len(data)

    for actual, expected in zip(spectra, data):
        assert actual.peaks == expected.peaks


def save_and_reload_spectra(filename, spectra: List[Spectrum]):
    """ Utility function to save spectra to msp and load them again.

    Params:
    -------
    spectra: Spectra objects to store

    Returns:
    --------
    reloaded_spectra: Spectra loaded from saved msp file.
    """

    save_as_msp(spectra, filename)
    reloaded_spectra = list(load_from_msp(filename))
    return reloaded_spectra


def test_num_peaks_last_metadata_field(filename, data):
    """ Test to check whether the last line before the peaks is NUM PEAKS: ... """
    save_as_msp(data, filename)

    with open(filename, mode='r') as file:
        content = file.readlines()
        for idx, line in enumerate(content):
            if line.startswith('NUM PEAKS: '):
                num_peaks = int(line.split()[2])
                peaks = content[idx + 1: idx + num_peaks + 1]
                for peak in peaks:
                    mz, intensity = peak.split()
                    mz = float(mz)
                    intensity = float(intensity)

                    assert isinstance(mz, float)
                    assert isinstance(intensity, float)
