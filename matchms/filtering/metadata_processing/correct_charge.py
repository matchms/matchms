import logging
from typing import Optional
import numpy as np
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def correct_charge(spectrum_in: SpectrumType, clone: Optional[bool] = True) -> Optional[SpectrumType]:
    """Correct charge values based on given ionmode.

    For some spectra, the charge value is either undefined or inconsistent with its
    ionmode, which is corrected by this filter.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with corrected charge derived from ionmode, or `None` if not present.
    """

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    ionmode = spectrum.get("ionmode", None)
    if ionmode:
        assert ionmode == ionmode.lower(), (
            "Ionmode field not harmonized.",
            "Apply 'make_ionmode_lowercase' filter first.",
        )

    charge = spectrum.get("charge", None)
    assert not isinstance(charge, str), ("Charge is given as string.", "Apply 'make_charge_int' filter first.")

    if charge is None:
        charge = 0

    if charge == 0 and ionmode == "positive":
        charge = 1
        logger.info("Guessed charge to 1 based on positive ionmode")
    elif charge == 0 and ionmode == "negative":
        charge = -1
        logger.info("Guessed charge to -1 based on negative ionmode")

    # Correct charge when in conflict with ionmode (trust ionmode more!)
    if np.sign(charge) == 1 and ionmode == "negative":
        logger.info("Changed sign of given charge: %s to match negative ionmode", charge)
        charge *= -1
    elif np.sign(charge) == -1 and ionmode == "positive":
        logger.warning("Changed sign of given charge: %s to match positive ionmode", charge)
        charge *= -1

    spectrum.set("charge", charge)

    return spectrum
