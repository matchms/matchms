import logging
from typing import Optional
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


class RequireCorrectIonmode(BaseSpectrumFilter):
    def __init__(self, ion_mode_to_keep):
        self.ion_mode_to_keep = ion_mode_to_keep

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        if self.ion_mode_to_keep not in {"positive", "negative", "both"}:
            raise ValueError("ion_mode_to_keep should be 'positive', 'negative' or 'both'")
        ion_mode = spectrum.get("ionmode")
        if self.ion_mode_to_keep == "both":
            if ion_mode in ("positive", "negative"):
                return spectrum

            logger.info("Spectrum was removed since ionmode was: %s which does not match positive or negative", ion_mode)
            return None
        if ion_mode == self.ion_mode_to_keep:
            return spectrum
        logger.info("Spectrum was removed since ionmode was: %s which does not match %s", ion_mode, self.ion_mode_to_keep)
        return None
