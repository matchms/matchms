import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import make_charge_int


@pytest.mark.parametrize("input_charge, corrected_charge", [('+1', 1),
                                                            ('1', 1),
                                                            (' 1 ', 1),
                                                            ('-2', -2),
                                                            ([-1, "stuff"], -1),
                                                            (['-3'], -3),
                                                            ('0', 0),
                                                            ('n/a', 'n/a'),
                                                            ('2+', 2),
                                                            ('2-', -2)])
def test_make_charge_int(input_charge, corrected_charge):
    """Test if example inputs are correctly converted"""
    spectrum_in = Spectrum(mz=numpy.array([100, 200.]),
                           intensities=numpy.array([0.7, 0.1]),
                           metadata={'charge': input_charge})

    spectrum = make_charge_int(spectrum_in)
    assert spectrum.get("charge") == corrected_charge, \
        "Expected different charge integer"


def test_empty_spectrum():
    spectrum_in = None
    spectrum = make_charge_int(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
