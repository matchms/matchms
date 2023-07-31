import numpy as np
import pytest
from matchms.filtering import make_charge_scalar
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("input_charge, corrected_charge", [
    ('+1', 1),
    ('1', 1),
    (' 1 ', 1),
    ('-1', -1),
    ([-1, "stuff"], -1)
])
def test_make_charge_scalar(input_charge, corrected_charge):
    """Test if example inputs are correctly converted"""
    mz = np.array([100, 200.])
    intensities = np.array([0.7, 0.1])
    metadata = {'charge': input_charge}
    spectrum_in = SpectrumBuilder().with_metadata(
        metadata).with_mz(mz).with_intensities(intensities).build()

    spectrum = make_charge_scalar(spectrum_in)
    assert spectrum.get("charge") == corrected_charge, "Expected different charge integer"


def test_empty_spectrum():
    spectrum_in = None
    spectrum = make_charge_scalar(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
