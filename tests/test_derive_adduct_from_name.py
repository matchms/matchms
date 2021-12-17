import numpy
import pytest
from testfixtures import LogCapture
from matchms import Spectrum
from matchms.filtering import derive_adduct_from_name
from matchms.logging_functions import reset_matchms_logger
from matchms.logging_functions import set_matchms_logger_level


@pytest.mark.parametrize("input_name, expected_adduct, expected_name", [
    ("peptideXYZ [M+H+K]", "[M+H+K]", "peptideXYZ"),
    ("GalCer(d18:2/16:1); [M+H]+", "[M+H]+", "GalCer(d18:2/16:1)")])
def test_derive_adduct_from_name(input_name, expected_adduct, expected_name):
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": input_name})
    spectrum = derive_adduct_from_name(spectrum_in)

    assert spectrum.get("adduct") == expected_adduct, "Expected different adduct."
    assert spectrum.get("compound_name") == expected_name, "Expected different cleaned name."


def test_derive_adduct_from_name_logging():
    set_matchms_logger_level("INFO")
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K]"})
    with LogCapture() as log:
        spectrum = derive_adduct_from_name(spectrum_in)

    assert spectrum.get("adduct") == "[M+H+K]", "Expected different adduct."
    assert spectrum.get("compound_name") == "peptideXYZ", "Expected different cleaned name."

    log.check(
        ('matchms', 'INFO', 'Removed adduct [M+H+K] from compound name.'),
        ('matchms', 'INFO', 'Added adduct [M+H+K] to metadata.')
    )
    reset_matchms_logger()


def test_derive_adduct_from_name_dont_overwrite_present_adduct():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K]",
                                     "adduct": "M+H"})

    spectrum = derive_adduct_from_name(spectrum_in)

    assert spectrum.get("adduct") == "M+H", "Expected different adduct."
    assert spectrum.get("compound_name") == "peptideXYZ", "Expected different cleaned name."


def test_derive_adduct_from_name_dont_remove_from_name():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K]"})

    spectrum = derive_adduct_from_name(spectrum_in, remove_adduct_from_name=False)

    assert spectrum.get("adduct") == "[M+H+K]", "Expected different adduct."
    assert spectrum.get("compound_name") == spectrum_in.get("compound_name"), "Expected no change to name."


def test_derive_adduct_from_name_no_compound_name_empty_name():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"name": ""})

    spectrum = derive_adduct_from_name(spectrum_in)

    assert spectrum.get("adduct", None) is None, "Expected None for adduct."
    assert spectrum.get("compound_name", None) is None, "Expected None for name."


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_adduct_from_name(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
