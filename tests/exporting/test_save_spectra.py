import os
import tempfile
import pytest
from matchms.exporting import save_spectra
from matchms.importing import load_from_mgf, load_spectra


def load_test_spectra_file():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "testdata", "testdata.mgf")
    spectra = list(load_from_mgf(spectrums_file))
    return spectra


@pytest.mark.parametrize("file_name",
                         ["spectra.msp",
                          "spectra.mgf",
                          "spectra.json",
                          "spectra.pickle"])
def test_spectra(file_name):
    """ Utility function to save spectra to msp and load them again.

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
