import numpy
import pytest
from testfixtures import LogCapture
from matchms import Spectrum
from matchms.filtering import add_compound_name


def test_add_compound_name_entry_name():
    """Test filled name field."""
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"name": "Testospectrum"})

    spectrum = add_compound_name(spectrum_in)

    assert spectrum.get("compound_name") == "Testospectrum", "Expected different compound name."


def test_add_compound_name_entry_title():
    """Test filled name field."""
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"title": "Testospectrum"})

    spectrum = add_compound_name(spectrum_in)

    assert spectrum.get("compound_name") == "Testospectrum", "Expected different compound name."


def test_add_compound_name_missing_entry():
    """Test filled name field."""
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"othername": "Testospectrum"})

    with LogCapture() as log:
        spectrum = add_compound_name(spectrum_in)

    assert spectrum.get("compound_name", None) is None, "Expected no compound name."
    log.check(
        ('matchms', 'WARNING', 'No compound name found in metadata.')
    )


def test_empty_spectrum():
    spectrum_in = None
    spectrum = add_compound_name(spectrum_in)

    assert spectrum is None, "Expected differnt handling of None spectrum."
