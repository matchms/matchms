import numpy as np
import pytest
from testfixtures import LogCapture
from matchms.filtering.require_precursor_mz import require_precursor_mz
from matchms.logging_functions import (reset_matchms_logger,
                                       set_matchms_logger_level)
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, expected", [
    [{"precursor_mz": 60.}, SpectrumBuilder().with_metadata(
        {"precursor_mz": 60}).build()],
    [{"precursor_mz": 0.0}, None],
    [{"precursor_mz": -3.5}, None],
    [{}, None]
])
def test_require_precursor_mz(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum == expected, "Expected no changes."


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": 100.}).build()

    spectrum = require_precursor_mz(spectrum_in)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), \
        "Expected input spectrum to remain unchanged."


def test_require_precursor_mz_with_wrong_precursor_mz():
    """Test if correct assert error is raised for precursor-mz as string."""
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": "445.0"}).build()

    with pytest.raises(AssertionError) as msg:
        _ = require_precursor_mz(spectrum_in)

    assert "Expected 'precursor_mz' to be a scalar number." in str(msg.value)


def test_require_precursor_mz_with_input_none():
    """Test if input spectrum is None."""
    spectrum_in = None
    spectrum = require_precursor_mz(spectrum_in)
    assert spectrum is None


@pytest.mark.parametrize("precursor_mz", [0, 10.0, -200])
def test_require_precursor_mz_fail_when_mz_too_small(precursor_mz):
    """Test if spectrum is None when precursor_mz <= minimum_accepted_mz"""
    set_matchms_logger_level("INFO")
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()
    spectrum_in.set("precursor_mz", precursor_mz)

    with LogCapture() as log:
        spectrum = require_precursor_mz(spectrum_in)

    assert spectrum is None, "Expected spectrum to be None."
    log.check(
        ('matchms', 'INFO', 'Spectrum without precursor_mz was set to None.')
    )
    reset_matchms_logger()


def test_require_precursor_mz_fail_because_below_zero():
    """Test if spectrum is None when precursor_mz < 0"""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()
    spectrum_in.set("precursor_mz", -3.5)

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum is None, "Expected spectrum to be None."
