import numpy as np
import pytest
from testfixtures import LogCapture
from matchms.filtering import add_losses
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("mz, loss_mz_to, expected_mz, expected_intensities", [
    [np.array([100, 150, 200, 300], dtype="float"), 1000, np.array([145, 245, 295, 345], "float"), np.array([1000, 100, 200, 700], "float")],
    [np.array([100, 150, 200, 450], dtype="float"), 1000, np.array([245, 295, 345], "float"), np.array([100, 200, 700], "float")],
    [np.array([100, 150, 200, 300], dtype="float"), 250, np.array([145, 245], "float"), np.array([1000, 100], "float")]
])
def test_add_losses_parameterized(mz, loss_mz_to, expected_mz, expected_intensities):
    intensities = np.array([700, 200, 100, 1000], "float")
    metadata = {"precursor_mz": 445.0}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(
        intensities).with_metadata(metadata).build()

    spectrum = add_losses(spectrum_in, loss_mz_to=loss_mz_to)

    assert np.allclose(spectrum.losses.mz, expected_mz), "Expected different loss m/z."
    assert np.allclose(spectrum.losses.intensities, expected_intensities), "Expected different intensities."


@pytest.mark.parametrize("mz, intensities", [
    [np.array([100, 150, 200, 300], dtype="float"), np.array([700, 200, 100, 1000], dtype="float")],
    [[], []]
])
def test_add_losses_without_precursor_mz_parameterized(mz, intensities):
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()
    spectrum = add_losses(spectrum_in)

    with LogCapture() as log:
        spectrum = add_losses(spectrum_in)

    assert spectrum == spectrum_in and spectrum is not spectrum_in
    log.check(
        ("matchms", "WARNING",
         "No precursor_mz found. Consider applying 'add_precursor_mz' filter first.")
    )


def test_add_losses_with_precursor_mz_wrong_type():
    """Test if correct assert error is raised for precursor-mz as string."""
    mz = np.array([100, 150, 200, 300], dtype="float")
    intensities = np.array([700, 200, 100, 1000], "float")
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
