import numpy as np
import pytest
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

    losses = spectrum_in.compute_losses(loss_mz_to=loss_mz_to)

    assert np.allclose(losses.mz, expected_mz), "Expected different loss m/z."
    assert np.allclose(losses.intensities, expected_intensities), "Expected different intensities."
