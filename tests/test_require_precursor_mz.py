import numpy
import pytest
from matchms import Spectrum
from matchms.filtering.require_precursor_mz import require_precursor_mz


def test_require_precursor_mz_pass():
    """Test with correct precursor mz present."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)
    spectrum_in.set("precursor_mz", 60.)

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum == spectrum_in, "Expected no changes."


def test_require_precursor_mz_fail_because_zero():
    """Test if spectrum is None when precursor_mz == 0"""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)
    spectrum_in.set("precursor_mz", 0.0)

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum is None, "Expected spectrum to be None."


def test_require_precursor_mz_fail_because_below_zero():
    """Test if spectrum is None when precursor_mz < 0"""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)
    spectrum_in.set("precursor_mz", -3.5)

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum is None, "Expected spectrum to be None."


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)
    spectrum_in.set("precursor_mz", 1.)

    spectrum = require_precursor_mz(spectrum_in)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), \
        "Expected input spectrum to remain unchanged."


def test_require_precursor_mz_without_precursor_mz():
    """Test if None is returned for missing precursor-mz."""
    spectrum_in = Spectrum(mz=numpy.array([10, 20, 30, 40], dtype="float"),
                           intensities=numpy.array([0, 1, 10, 100],
                                                   dtype="float"),
                           metadata={})

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum is None, "Expected spectrum to be None."


def test_require_precursor_mz_with_wrong_precursor_mz():
    """Test if correct assert error is raised for precursor-mz as string."""
    spectrum_in = Spectrum(mz=numpy.array([10, 20, 30, 40], dtype="float"),
                           intensities=numpy.array([0, 1, 10, 100],
                                                   dtype="float"),
                           metadata={"precursor_mz": "445.0"})

    with pytest.raises(AssertionError) as msg:
        _ = require_precursor_mz(spectrum_in)

    assert "Expected 'precursor_mz' to be a scalar number." in str(msg.value)


def test_require_precursor_mz_with_input_none():
    """Test if input spectrum is None."""
    spectrum_in = None
    spectrum = require_precursor_mz(spectrum_in)
    assert spectrum is None
