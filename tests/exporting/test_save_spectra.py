import logging
import os
import re
import tempfile
import pytest
from matchms.exporting.save_spectra import save_as_pickled_file, save_spectra
from matchms.importing import load_from_mgf, load_spectra


def load_test_spectra_file():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", "testdata.mgf")
    spectra = list(load_from_mgf(spectra_file))
    return spectra


@pytest.mark.parametrize("file_name", ["spectra.msp", "spectra.mgf", "spectra.json", "spectra.pickle"])
def test_spectra(file_name, caplog):
    """Utility function to save spectra to msp and load them again.

    Params:
    -------
    spectra: Spectra objects to store

    Returns:
    --------
    reloaded_spectra: Spectra loaded from saved msp file.
    """
    spectrum_list = load_test_spectra_file()
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, file_name)
        save_spectra(spectrum_list, filename)
        assert os.path.exists(filename)
        reloaded_spectra = list(load_spectra(filename))
    assert len(reloaded_spectra) == len(spectrum_list)
    for i, spectrum in enumerate(spectrum_list):
        reloaded_spectrum = reloaded_spectra[i]
        # Num_peaks is sometimes added during saving. So to be able to compare it is set to None
        spectrum.set("num_peaks", None)
        reloaded_spectrum.set("num_peaks", None)
        assert spectrum == reloaded_spectrum

    # Test file exists error
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, file_name)

        with open(filename, "w", encoding="utf-8") as file:
            file.write("content")
        assert os.path.exists(filename)

        with pytest.raises(FileExistsError, match=re.escape(f"The specified file: {filename} already exists.")):
            save_spectra(spectrum_list, filename)

    # Test append to different filetype not supported error
    ftype = os.path.splitext(file_name)[1].lower()[1:]
    if ftype in ["json", "pickle"]:
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, file_name)

            with pytest.raises(ValueError, match=re.escape(f"{ftype} isn't supported for when `append` is True")):
                save_spectra(spectra=spectrum_list, file=filename, append=True)

    # Test logger warning when using pickle
    if ftype == "pickle":
        with caplog.at_level(logging.ERROR):
            with tempfile.TemporaryDirectory() as temp_dir:
                filename = os.path.join(temp_dir, file_name)
                save_spectra(spectra=spectrum_list, file=filename, export_style="invalid")

        assert "The only available export style for pickle is 'matchms', your export style invalid" in caplog.text


def test_spectra_invalid_ext():
    spectrum_list = load_test_spectra_file()

    # Test invalid file ext
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "invalid.txt")

        with pytest.raises(TypeError, match=re.escape(f"File extension of file: {filename} is not recognized")):
            save_spectra(spectrum_list, filename)


@pytest.mark.parametrize("file_name", ["spectra.pickle"])
def test_save_as_pickled_file_none_spectra(file_name):
    """Tests only pickled file saving with filtered None valued spectra

    Params:
    -------
    spectra: Spectra objects to store

    Returns:
    --------
    reloaded_spectra: Spectra loaded from saved msp file.
    """
    spectrum_list = load_test_spectra_file()

    # Test file exists error
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, file_name)

        with open(filename, "w", encoding="utf-8") as file:
            file.write("content")
        assert os.path.exists(filename)

        with pytest.raises(FileExistsError, match=re.escape(f"The file '{filename}' already exists.")):
            save_as_pickled_file(spectrum_list, filename)

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, file_name)

        # Test no spectra list
        with pytest.raises(TypeError, match="Expected list of spectra"):
            save_as_pickled_file("invalid", filename)

        # Test empty spectra list
        spectrum_list = [None]
        with pytest.raises(TypeError, match="Expected list of spectra"):
            save_as_pickled_file(spectrum_list, filename)
