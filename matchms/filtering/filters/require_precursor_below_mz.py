import logging
from matchms.typing import SpectrumType
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter

logger = logging.getLogger("matchms")


class RequirePrecursorBelowMz(BaseSpectrumFilter):
    def __init__(self, max_mz: float = 1000):
        self.max_mz = max_mz

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        precursor_mz = spectrum.get("precursor_mz", None)
        assert precursor_mz is not None, "Precursor mz absent."
        assert isinstance(precursor_mz, (float, int)), ("Expected 'precursor_mz' to be a scalar number.",
                                                        "Consider applying 'add_precursor_mz' filter first.")
        assert self.max_mz >= 0, "max_mz must be a positive scalar."
        if precursor_mz >= self.max_mz:
            logger.info("Spectrum with precursor_mz %s (>%s) was set to None.",
                        str(precursor_mz), str(self.max_mz))
            return None

        return spectrum
