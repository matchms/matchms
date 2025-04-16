import numpy as np
import pytest
from matchms.filtering import make_charge_int
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "input_charge, corrected_charge",
    [
        ("+1", 1),
        ("1", 1),
        (" 1 ", 1),
        ("-2", -2),
        ([-1, "stuff"], -1),
        (["-3"], -3),
        ("0", 0),
        ("n/a", "n/a"),
        ("2+", 2),
        ("2-", -2),
    ],
)
def test_make_charge_int(input_charge, corrected_charge):
    """Test if example inputs are correctly converted"""
    mz = np.array([100, 200.0])
    intensities = np.array([0.7, 0.1])
    metadata = {"charge": input_charge}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = make_charge_int(spectrum_in)
    assert spectrum.get("charge") == corrected_charge, "Expected different charge integer"


def test_empty_spectrum():
    spectrum_in = None
    spectrum = make_charge_int(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
