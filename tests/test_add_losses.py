import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import add_losses
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("mz, loss_mz_to, expected_mz, expected_intensities", [
    [numpy.array([100, 150, 200, 300], dtype="float"), 1000, numpy.array([145, 245, 295, 345], "float"), numpy.array([1000, 100, 200, 700], "float")],
    [numpy.array([100, 150, 200, 450], dtype="float"), 1000, numpy.array([245, 295, 345], "float"), numpy.array([100, 200, 700], "float")],
    [numpy.array([100, 150, 200, 300], dtype="float"), 250, numpy.array([145, 245], "float"), numpy.array([1000, 100], "float")]
])
def test_add_losses_parameterized(mz, loss_mz_to, expected_mz, expected_intensities):
    intensities = numpy.array([700, 200, 100, 1000], "float")
    metadata = {"precursor_mz": 445.0}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(
        intensities).with_metadata(metadata).build()

    spectrum = add_losses(spectrum_in, loss_mz_to=loss_mz_to)

    assert numpy.allclose(spectrum.losses.mz, expected_mz), "Expected different loss m/z."
    assert numpy.allclose(spectrum.losses.intensities, expected_intensities), "Expected different intensities."


@pytest.mark.parametrize("mz, intensities", [
    [numpy.array([100, 150, 200, 300], dtype="float"), numpy.array([700, 200, 100, 1000], dtype="float")],
    [[], []]
])
def test_add_losses_without_precursor_mz_parameterized(mz, intensities):
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()
    spectrum = add_losses(spectrum_in)

    assert spectrum == spectrum_in and spectrum is not spectrum_in


def test_add_losses_with_precursor_mz_wrong_type():
    """Test if correct assert error is raised for precursor-mz as string."""
    mz=numpy.array([100, 150, 200, 300], dtype="float")
    intensities = numpy.array([700, 200, 100, 1000], "float")
    metadata = {"precursor_mz": "445.0"}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(
        intensities).with_metadata(metadata).build()

    with pytest.raises(AssertionError) as msg:
        _ = add_losses(spectrum_in)

    assert "Expected 'precursor_mz' to be a scalar number." in str(msg.value)



def test_add_losses_with_input_none():
    """Test if input spectrum is None."""
    spectrum_in = None
    spectrum = add_losses(spectrum_in)
    assert spectrum is None





