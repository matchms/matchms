import numpy as np
from matchms.filtering.filters.select_by_template import SelectByTemplate
from matchms.typing import SpectrumType


class SelectByMz(SelectByTemplate):
    def __init__(self, mz_from: float, mz_to: float):
        assert mz_from <= mz_to, "'mz_from' should be smaller than or equal to 'mz_to'."
        self.mz_from = mz_from
        self.mz_to = mz_to

    def _create_condition(self, spectrum: SpectrumType):
        return np.logical_and(
            self.mz_from <= spectrum.peaks.mz,
            spectrum.peaks.mz <= self.mz_to
        )