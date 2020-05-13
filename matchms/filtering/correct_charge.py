import numpy
from ..typing import SpectrumType


def correct_charge(spectrum_in: SpectrumType) -> SpectrumType:
    """
    For some spectrums, the charge value is either undefined or inconsistent with its
    ionmode, which is corrected by this filter.
    """

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    ionmode = spectrum.get("ionmode", None)

    charge = spectrum.get("charge", None)

    if charge is None:
        charge = 0
    elif charge == 0 and ionmode == 'positive':
        charge = 1
    elif charge == 0 and ionmode == 'negative':
        charge = -1
    else:
        pass

    # Correct charge when in conflict with ionmode (trust ionmode more!)
    if numpy.sign(charge) == 1 and ionmode == 'negative':
        charge *= -1
    elif numpy.sign(charge) == -1 and ionmode == 'positive':
        charge *= -1

    spectrum.set("charge", charge)

    return spectrum
