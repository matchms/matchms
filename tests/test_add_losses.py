import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import add_losses


def test_add_losses():
    """Test if all losses are correctly generated form mz values and precursor-m/z."""
    spectrum_in = Spectrum(mz=numpy.array([100, 150, 200, 300], dtype="float"),
                           intensities=numpy.array([700, 200, 100, 1000], dtype="float"),
                           metadata={"precursor_mz": 445.0})

    spectrum = add_losses(spectrum_in)

    expected_mz = numpy.array([145, 245, 295, 345], "float")
    expected_intensities = numpy.array([1000, 100, 200, 700], "float")
    assert numpy.allclose(spectrum.losses.mz, expected_mz), "Expected different loss m/z."
    assert numpy.allclose(spectrum.losses.intensities, expected_intensities), "Expected different intensities."


def test_add_losses_without_precursor_mz():
    """Test if no changes are done without having a precursor-m/z."""
    spectrum_in = Spectrum(mz=numpy.array([100, 150, 200, 300], dtype="float"),
                           intensities=numpy.array([700, 200, 100, 1000], dtype="float"))

    spectrum = add_losses(spectrum_in)

    assert spectrum == spectrum_in and spectrum is not spectrum_in


def test_add_losses_with_precursor_mz_wrong_type():
    """Test if correct assert error is raised for precursor-mz as string."""
    spectrum_in = Spectrum(mz=numpy.array([100, 150, 200, 300], dtype="float"),
                           intensities=numpy.array([700, 200, 100, 1000], dtype="float"),
                           metadata={"precursor_mz": "445.0"})

    with pytest.raises(AssertionError) as msg:
        _ = add_losses(spectrum_in)

    assert "Expected 'precursor_mz' to be a scalar number." in str(msg.value)


def test_add_losses_returns_new_spectrum_instance():
    """Test if no change is done to empty spectrum."""
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"))

    spectrum = add_losses(spectrum_in)

    assert spectrum == spectrum_in and spectrum is not spectrum_in


def test_add_losses_with_input_none():
    """Test if input spectrum is None."""
    spectrum_in = None
    spectrum = add_losses(spectrum_in)
    assert spectrum is None


def test_add_losses_with_peakmz_larger_precursormz():
    """Test if losses are correctly generated and loss < 0 is discarded."""
    spectrum_in = Spectrum(mz=numpy.array([100, 150, 200, 450], dtype="float"),
                           intensities=numpy.array([700, 200, 100, 1000], dtype="float"),
                           metadata={"precursor_mz": 445.0})

    spectrum = add_losses(spectrum_in)

    expected_mz = numpy.array([245, 295, 345], "float")
    expected_intensities = numpy.array([100, 200, 700], "float")
    assert numpy.allclose(spectrum.losses.mz, expected_mz), "Expected different loss m/z."
    assert numpy.allclose(spectrum.losses.intensities, expected_intensities), "Expected different intensities."


def test_add_losses_with_max_loss_mz_250():
    """Test if losses are correctly generated and losses with mz > 250 are discarded."""
    spectrum_in = Spectrum(mz=numpy.array([100, 150, 200, 300], dtype="float"),
                           intensities=numpy.array([700, 200, 100, 1000], dtype="float"),
                           metadata={"precursor_mz": 445.0})

    spectrum = add_losses(spectrum_in, loss_mz_to=250)

    expected_mz = numpy.array([145, 245], "float")
    expected_intensities = numpy.array([1000, 100], "float")
    assert numpy.allclose(spectrum.losses.mz, expected_mz), "Expected different loss m/z."
    assert numpy.allclose(spectrum.losses.intensities, expected_intensities), "Expected different intensities."
