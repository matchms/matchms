import logging
from math import ceil
from typing import Optional
from matchms.typing import SpectrumType
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter


logger = logging.getLogger("matchms")


class RequireMinimumNumberOfPeaks(BaseSpectrumFilter):
    def __init__(self, n_required: int = 10, ratio_required: Optional[float] = None):
        self.n_required = n_required
        self.ratio_required = ratio_required

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        parent_mass = spectrum.get("parent_mass", None)
        if parent_mass and self.ratio_required:
            n_required_by_mass = int(ceil(self.ratio_required * parent_mass))
            threshold = max(self.n_required, n_required_by_mass)
        else:
            threshold = self.n_required

        if spectrum.peaks.intensities.size < threshold:
            logger.info("Spectrum with %s (<%s) peaks was set to None.",
                        str(spectrum.peaks.intensities.size), str(threshold))
            return None

        return spectrum
