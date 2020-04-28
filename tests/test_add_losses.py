from matchms import Spectrum
from matchms.filtering import add_losses
import numpy
import pytest


def test_add_losses():
    spectrum_in = Spectrum(mz=numpy.array([100, 150, 200, 300], dtype="float"),
                           intensities=numpy.array([700, 200, 100, 1000], dtype="float"),
                           metadata={"precursor_mz": 445.0})

    spectrum = add_losses(spectrum_in)

    assert numpy.allclose(spectrum.losses.mz, numpy.array([145, 245, 295, 345], "float"))


def test_add_losses_without_precursor_mz():
    spectrum_in = Spectrum(mz=numpy.array([100, 150, 200, 300], dtype="float"),
                           intensities=numpy.array([700, 200, 100, 1000], dtype="float"))

    spectrum = add_losses(spectrum_in)

    assert spectrum == spectrum_in and spectrum is not spectrum_in


def test_add_losses_with_precursor_mz_wrong_type():

    spectrum_in = Spectrum(mz=numpy.array([100, 150, 200, 300], dtype="float"),
                           intensities=numpy.array([700, 200, 100, 1000], dtype="float"),
                           metadata={"precursor_mz": "445.0"})

    with pytest.raises(AssertionError) as msg:
        _ = add_losses(spectrum_in)

    assert str(msg.value) == "Expected 'precursor_mz' to be a scalar number."


def test_add_losses_returns_new_spectrum_instance():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"))

    spectrum = add_losses(spectrum_in)

    assert spectrum == spectrum_in and spectrum is not spectrum_in


def test_add_losses_with_input_none():
    spectrum_in = None
    spectrum = add_losses(spectrum_in)
    assert spectrum is None
