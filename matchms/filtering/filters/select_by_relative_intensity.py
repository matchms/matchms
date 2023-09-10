import numpy as np
from matchms.filtering.filters.select_by_template import SelectByTemplate
from matchms.typing import SpectrumType


class SelectByRelativeIntensity(SelectByTemplate):
    def __init__(self, intensity_from: float, intensity_to: float):
        assert intensity_from >= 0.0, "'intensity_from' should be larger than or equal to 0."
        assert intensity_to <= 1.0, "'intensity_to' should be smaller than or equal to 1.0."
        assert intensity_from <= intensity_to, "'intensity_from' should be smaller than or equal to 'intensity_to'."
        self.intensity_from = intensity_from
        self.intensity_to = intensity_to

    def _create_condition(self, spectrum: SpectrumType):
        if len(spectrum.peaks) > 0:
            scale_factor = np.max(spectrum.peaks.intensities)
            normalized_intensities = spectrum.peaks.intensities / scale_factor
            return np.logical_and(
                self.intensity_from <= normalized_intensities,
                normalized_intensities <= self.intensity_to
            )
        else:
            return np.array([], dtype=bool)