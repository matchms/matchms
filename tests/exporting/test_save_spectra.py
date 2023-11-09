import os
import tempfile
from typing import List
import numpy as np
import pytest
from matchms import Spectrum
from matchms.exporting import save_spectra
from matchms.importing import load_spectra
from ..builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectrum_list():
    mz = np.array([100, 200, 290, 490, 510], dtype="float")
    intensities = np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float")
    spectrum = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()
    return [spectrum]


@pytest.mark.parametrize("file_name",
                         ["spectra.msp",
                          "spectra.mgf",
                          "spectra.json",
                          "spectra.pickle"])
def test_spectra(file_name, spectrum_list: List[Spectrum]):
    """ Utility function to save spectra to msp and load them again.

    Params:
    -------
    spectra: Spectra objects to store

    Returns:
    --------
    reloaded_spectra: Spectra loaded from saved msp file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, file_name)
    save_spectra(spectrum_list, filename)
    reloaded_spectra = list(load_spectra(filename))
    assert save_spectra == reloaded_spectra


# def test_wrong_filename_exception():
#     """ Test for exception being thrown if output file doesn't end with .msp. """
#     with tempfile.TemporaryDirectory() as temp_dir:
#         filename = os.path.join(temp_dir, "test.mzml")
#
#         with pytest.raises(AssertionError) as exception:
#             save_as_msp(None, filename)
#
#         message = exception.value.args[0]
#         assert message == "File extension '.mzml' not allowed."
