import numpy as np
import pytest
from matchms.filtering import add_precursor_mz
from matchms.logging_functions import reset_matchms_logger, set_matchms_logger_level
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, expected", [[{"precursor_mz": 444.0}, 444.0], [{}, None], [{"pepmass": (444.0, 10)}, 444.0]])
def test_add_precursor_mz(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum.get("precursor_mz") == expected, "Expected different precursor_mz."


def test_add_precursor_mz_only_pepmass_present(caplog):
    """Test if precursor_mz is correctly derived if only pepmass is present."""
    set_matchms_logger_level("INFO")
    mz = np.array([], dtype="float")
    intensities = np.array([], dtype="float")
    metadata = {"pepmass": (444.0, 10)}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).with_mz(mz).with_intensities(intensities).build()

    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum.get("precursor_mz") == 444.0, "Expected different precursor_mz."
    assert "Added precursor_mz entry based on field 'pepmass'" in caplog.text, "Expected different log message"
    reset_matchms_logger()


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ["precursor_mz", "444.0", 444.0],
        ["precursormz", "15.6", 15.6],
        ["precursormz", 15.0, 15.0],
        ["precursor_mass", "17.887654", 17.887654],
        ["precursor_mass", "N/A", None],
        ["precursor_mass", "test", None],
        ["pepmass", (33.89, 50), 33.89],
        ["pepmass", "None", None],
        ["pepmass", None, None],
    ],
)
def test_add_precursor_mz_no_precursor_mz(key, value, expected):
    """Test if precursor_mz is correctly derived if "precursor_mz" is str."""
    mz = np.array([], dtype="float")
    intensities = np.array([], dtype="float")
    metadata = {key: value}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).with_mz(mz).with_intensities(intensities).build()

    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum.get("precursor_mz") == expected, "Expected different precursor_mz."


@pytest.mark.parametrize(
    "key, value, expected_log",
    [
        ["precursor_mz", "N/A", "No precursor_mz found in metadata."],
        ["precursor_mass", "test", "test can't be converted to float."],
        ["precursor_mz", None, "No precursor_mz found in metadata."],
        ["pepmass", None, "No precursor_mz found in metadata."],
        ["precursor_mz", [], "Found precursor_mz of undefined type."],
    ],
)
def test_add_precursor_mz_logging(key, value, expected_log, caplog):
    """Test if precursor_mz is correctly derived if "precursor_mz" is str."""
    mz = np.array([], dtype="float")
    intensities = np.array([], dtype="float")
    metadata = {key: value}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).with_mz(mz).with_intensities(intensities).build()

    _ = add_precursor_mz(spectrum_in)

    assert expected_log in caplog.text, "Expected different logging message."


def test_empty_spectrum():
    spectrum_in = None
    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
