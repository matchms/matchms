import logging
import numpy as np
from matchms.typing import SpectrumType
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter


logger = logging.getLogger("matchms")


class CorrectCharge(BaseSpectrumFilter):
    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
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
