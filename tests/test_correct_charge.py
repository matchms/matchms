import numpy
from matchms import Spectrum
from matchms.filtering import correct_charge


def test_correct_charge_no_ionmode():
    """Test if no charge is added for empty ionmode."""
    spectrum_in = Spectrum(mz=numpy.array([], dtype='float'),
                           intensities=numpy.array([], dtype='float'),
                           metadata={})

    spectrum = correct_charge(spectrum_in)

    assert spectrum.get("charge") == 0, "Expected zero charge value."


def test_correct_charge_add_charge():
    """Test if charge is corrected as expected."""
    spectrum_in = Spectrum(mz=numpy.array([], dtype='float'),
                           intensities=numpy.array([], dtype='float'),
                           metadata={"ionmode": "positive"})

    spectrum = correct_charge(spectrum_in)

    assert spectrum.get("charge") == 1, "Expected different charge value."


def test_correct_charge_correct_charge_sign_plus_to_min():
    """Test if charge is corrected as expected."""
    spectrum_in = Spectrum(mz=numpy.array([], dtype='float'),
                           intensities=numpy.array([], dtype='float'),
                           metadata={"ionmode": "negative",
                                     "charge": 2})

    spectrum = correct_charge(spectrum_in)

    assert spectrum.get("charge") == -2, "Expected different charge value."


def test_correct_charge_correct_charge_sign_min_to_plus():
    """Test if charge is corrected as expected."""
    spectrum_in = Spectrum(mz=numpy.array([], dtype='float'),
                           intensities=numpy.array([], dtype='float'),
                           metadata={"ionmode": "positive",
                                     "charge": -2})

    spectrum = correct_charge(spectrum_in)

    assert spectrum.get("charge") == 2, "Expected different charge value."


def test_correct_charge_empty_spectrum():
    spectrum_in = None
    spectrum = correct_charge(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
