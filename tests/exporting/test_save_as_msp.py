import logging
import os
import tempfile
from typing import List
import numpy as np
import pytest
from matchms import Spectrum
from matchms.exporting import save_as_msp
from matchms.importing import load_from_mgf, load_from_msp
from ..builder_Spectrum import SpectrumBuilder


@pytest.fixture
def none_spectrum():
    return None


@pytest.fixture
def spectrum():
    mz = np.array([100, 200, 290, 490, 510], dtype="float")
    intensities = np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float")
    return SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()


@pytest.fixture(params=["rcx_gc-ei_ms_20201028_perylene.msp", "MoNA-export-GC-MS-first10.msp", "Hydrogen_chloride.msp"])
def data(request):
    spectra = load_test_spectra_file(request.param)
    return spectra


def load_test_spectra_file(test_filename):
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", test_filename)
    spectra = list(load_from_msp(spectra_file))
    return spectra


@pytest.fixture
def filename():
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "test.msp")
        yield filename


def save_and_reload_spectra(filename, spectra: List[Spectrum], write_peak_comments=True):
    """Utility function to save spectra to msp and load them again.

    Params:
    -------
    spectra: Spectra objects to store

    Returns:
    --------
    reloaded_spectra: Spectra loaded from saved msp file.
    """

    save_as_msp(spectra, filename, write_peak_comments)
    reloaded_spectra = list(load_from_msp(filename))
    return reloaded_spectra


def test_wrong_filename_exception(caplog):
    """Test for exception being thrown if output file doesn't end with .msp."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "test.mzml")

        with pytest.raises(AssertionError) as exception:
            save_as_msp(None, filename)

        message = exception.value.args[0]
        assert message == "File extension '.mzml' not allowed."

        # Test warning log for filename except extensions not allowed
        filename = os.path.join(temp_dir, "test.txt")

        with caplog.at_level(logging.WARNING):
            save_as_msp(None, filename)

        assert "Spectrum(s) will be stored as msp file with extension .txt" in caplog.text


# Using tmp_path fixture from pytest: https://docs.pytest.org/en/stable/tmpdir.html#the-tmp-path-fixture
def test_file_exists_single_spectrum(spectrum, filename):
    """Test checking if the file is created."""
    save_as_msp(spectrum, filename)
    assert os.path.isfile(filename)


def test_name_comes_first(spectrum: Spectrum, filename: str):
    spectrum.set("ionization", "positive")
    spectrum.set("compound_name", "test")
    save_as_msp(spectrum, filename)

    with open(filename, "r", encoding="UTF-8") as file:
        assert file.readline() == "COMPOUND_NAME: test\n"


def test_peak_sep(spectrum: Spectrum, filename: str):
    save_as_msp(spectrum, filename, peak_sep="  ")

    with open(filename, "r", encoding="UTF-8") as file:
        lines = file.readlines()
        assert "100.0  0.1\n" in lines


def test_stores_all_spectra(filename, data):
    """Test checking if all spectra contained in the original file are stored
    and loaded back in properly."""
    spectra = save_and_reload_spectra(filename, data)

    assert len(spectra) == len(data)


def test_have_metadata(filename, data):
    """Test checking of all metadate is stored correctly."""
    spectra = save_and_reload_spectra(filename, data)

    assert len(spectra) == len(data)

    for actual, expected in zip(spectra, data):
        assert actual.metadata == expected.metadata


def test_have_peaks(filename, data):
    """Test checking if all peaks are stored correctly."""
    spectra = save_and_reload_spectra(filename, data)

    assert len(spectra) == len(data)

    for actual, expected in zip(spectra, data):
        assert actual.peaks == expected.peaks


def test_dont_write_peak_comments(filename, data):
    """Test checking if no peak comments are written to file."""
    spectra = save_and_reload_spectra(filename, data, write_peak_comments=False)

    assert len(spectra) == len(data)

    for actual, _ in zip(spectra, data):
        assert actual.peak_comments is None, "Expected that no peak comments are written to file"


def test_num_peaks_last_metadata_field(filename, data):
    """Test to check whether the last line before the peaks is NUM PEAKS: ..."""
    save_as_msp(data, filename)

    with open(filename, mode="r", encoding="utf-8") as file:
        content = file.readlines()
        for idx, line in enumerate(content):
            if line.startswith("NUM PEAKS: "):
                num_peaks = int(line.split()[2])
                peaks = content[idx + 1 : idx + num_peaks + 1]
                for peak in peaks:
                    mz, intensity = peak.split()[:2]
                    mz = float(mz)
                    intensity = float(intensity)

                    assert isinstance(mz, float)
                    assert isinstance(intensity, float)


@pytest.mark.parametrize("test_file", ["MoNA-export-GC-MS-first10.msp", "massbank_five_spectra.msp"])
def test_write_append(test_file, filename):
    expected = load_test_spectra_file(test_file)
    save_as_msp(expected[:2], filename, mode="a")
    save_as_msp(expected[2:], filename, mode="a")

    actual = list(load_from_msp(filename))

    assert expected == actual


@pytest.mark.parametrize(
    "test_file, expected_file, style", [["massbank_five_spectra.msp", "riken_style_five_spectra.msp", "riken"]]
)
def test_save_as_msp_export_style(test_file, expected_file, style, filename):
    expected = load_test_spectra_file(expected_file)
    data = load_test_spectra_file(test_file)
    save_as_msp(data, filename, mode="w", style=style)
    actual = list(load_from_msp(filename))
    assert expected == actual


@pytest.mark.parametrize("test_file, expected_file", [["save_as_msp_from_mgf.mgf", "save_as_msp_from_mgf.msp"]])
def test_save_as_msp_from_mgf(test_file, expected_file, filename):
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", test_file)
    actual = list(load_from_mgf(spectra_file))
    expected = load_test_spectra_file(expected_file)
    save_as_msp(actual, filename, mode="w", write_peak_comments=True)
    actual = list(load_from_msp(filename))
    assert expected == actual


def test_filter_none_spectra(filename, data):
    """Test for removal of None valued Spectra"""

    spectra = data
    spectra.append(None)

    save_as_msp(spectra, filename)
    reloaded_spectra = list(load_from_msp(filename))

    assert len(reloaded_spectra) == (len(spectra) - 1)
