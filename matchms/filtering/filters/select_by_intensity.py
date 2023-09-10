import numpy as np
from matchms.filtering.filters.select_by_template import SelectByTemplate
from matchms.typing import SpectrumType


class SelectByIntensity(SelectByTemplate):
    def __init__(self, intensity_from: float, intensity_to: float):
        assert intensity_from <= intensity_to, "'intensity_from' should be smaller than or equal to 'intensity_to'."
        self.intensity_from = intensity_from
        self.intensity_to = intensity_to

    def _create_condition(self, spectrum: SpectrumType):
        return np.logical_and(
            self.intensity_from <= spectrum.peaks.intensities,
            spectrum.peaks.intensities <= self.intensity_to
        )