import numpy
import pytest
from testfixtures import LogCapture
from matchms.filtering import require_precursor_below_mz
from matchms.logging_functions import reset_matchms_logger
from matchms.logging_functions import set_matchms_logger_level
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, max_mz, expected", [
    [{"precursor_mz": 60.}, 1000, SpectrumBuilder().with_metadata(
        {"precursor_mz": 60}).build()],
    [{"precursor_mz": 60.}, 50, None]
])
def test_require_precursor_below_mz(metadata, max_mz, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = require_precursor_below_mz(spectrum_in, max_mz=max_mz)
    assert spectrum == expected


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": 1.}).build()

    spectrum = require_precursor_below_mz(spectrum_in)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), "Expected input spectrum to remain unchanged."


def test_require_precursor_below_without_precursor_mz():
    """Test if correct assert error is raised for missing precursor-mz."""
    spectrum_in = SpectrumBuilder().build()

    with pytest.raises(AssertionError) as msg:
        _ = require_precursor_below_mz(spectrum_in)

    assert str(msg.value) == "Precursor mz absent.", "Expected different error message."


def test_require_precursor_below_with_wrong_precursor_mz():
    """Test if correct assert error is raised for precursor-mz as string."""
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": "445.0"}).build()

    with pytest.raises(AssertionError) as msg:
        _ = require_precursor_below_mz(spectrum_in)

    assert "Expected 'precursor_mz' to be a scalar number." in str(msg.value)


def test_with_input_none():
    """Test if input spectrum is None."""
    spectrum_in = None
    spectrum = require_precursor_below_mz(spectrum_in)
    assert spectrum is None


def test_require_precursor_below_mz_no_params():
    """Using default parameterse with precursor mz present."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()
    spectrum_in.set("precursor_mz", 60.)

    spectrum = require_precursor_below_mz(spectrum_in)

    assert spectrum == spectrum_in, "Expected no changes."


def test_require_precursor_below_mz_max_50():
    """Set max_mz to 50."""
    set_matchms_logger_level("INFO")
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()
    spectrum_in.set("precursor_mz", 60.)

    with LogCapture() as log:
        spectrum = require_precursor_below_mz(spectrum_in, max_mz=50)

    assert spectrum is None, "Expected spectrum to be None."
    log.check(
        ('matchms', 'INFO', 'Spectrum with precursor_mz 60.0 (>50) was set to None.')
    )
    reset_matchms_logger()
