import logging
import numpy as np
from ..typing import SpectrumType


logger = logging.getLogger("matchms")


def correct_charge(spectrum_in: SpectrumType) -> SpectrumType:
    """Correct charge values based on given ionmode.

    For some spectrums, the charge value is either undefined or inconsistent with its
    ionmode, which is corrected by this filter.

    Parameters
    ----------
    spectrum_in
        Input spectrum.
    """

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    ionmode = spectrum.get("ionmode", None)
    if ionmode:
        assert ionmode == ionmode.lower(), ("Ionmode field not harmonized.",
                                            "Apply 'make_ionmode_lowercase' filter first.")

    charge = spectrum.get("charge", None)
    assert not isinstance(charge, str), ("Charge is given as string.",
                                         "Apply 'make_charge_int' filter first.")

    if charge is None:
        charge = 0

    if charge == 0 and ionmode == 'positive':
        charge = 1
        logger.info("Guessed charge to 1 based on positive ionmode")
    elif charge == 0 and ionmode == 'negative':
        charge = -1
        logger.info("Guessed charge to -1 based on negative ionmode")

    # Correct charge when in conflict with ionmode (trust ionmode more!)
    if np.sign(charge) == 1 and ionmode == 'negative':
        charge *= -1
        logger.warning("Changed sign of given charge to match negative ionmode")
    elif np.sign(charge) == -1 and ionmode == 'positive':
        charge *= -1
        logger.warning("Changed sign of given charge to match positive ionmode")

    spectrum.set("charge", charge)

    return spectrum
