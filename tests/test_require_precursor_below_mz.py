import numpy
from matchms import Spectrum
from matchms.filtering import require_precursor_below_mz


def test_require_precursor_below_mz_no_params_or_precursor():
    """No parameters provided and no precursor mz present."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_precursor_below_mz(spectrum_in)

    assert spectrum == spectrum_in, "Expected no changes."


def test_require_precursor_below_mz_no_params():
    """No parameters provided but precursor mz present."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)
    spectrum_in.set("precursor_mz", 60.)

    spectrum = require_precursor_below_mz(spectrum_in)

    assert spectrum == spectrum_in, "Expected no changes."


def test_require_precursor_below_mz_max_50():
    """Set max_mz to 50."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)
    spectrum_in.set("precursor_mz", 60.)

    spectrum = require_precursor_below_mz(spectrum_in, max_mz = 50)

    assert spectrum is None, "Expected spectrum to be None."
