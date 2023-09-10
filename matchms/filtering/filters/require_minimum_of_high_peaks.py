import logging
from matchms.typing import SpectrumType
#from .select_by_relative_intensity import select_by_relative_intensity
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.filtering.filters.select_by_relative_intensity import SelectByRelativeIntensity


logger = logging.getLogger("matchms")


class RequireMinimumOfHighPeaks(BaseSpectrumFilter):
    def __init__(self, no_peaks: int = 5, intensity_percent: float = 2.0):
        self.no_peaks = no_peaks
        self.intensity_percent = intensity_percent

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        assert self.no_peaks >= 1, "no_peaks must be a positive nonzero integer."
        assert 0 <= self.intensity_percent <= 100, "intensity_percent must be a scalar between 0-100."
        intensities_above_p = SelectByRelativeIntensity(intensity_from=self.intensity_percent/100, intensity_to=1.0).process(spectrum)
        if len(intensities_above_p.peaks) < self.no_peaks:
            logger.info("Spectrum with %s (<%s) peaks was set to None.",
                        str(len(intensities_above_p.peaks)), str(self.no_peaks))
            return None

        return spectrum
