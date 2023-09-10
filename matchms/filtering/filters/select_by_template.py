from matchms.Fragments import Fragments
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.typing import SpectrumType


class SelectByTemplate(BaseSpectrumFilter):
    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        condition = self._create_condition(spectrum)
        spectrum.peaks = Fragments(
            mz=spectrum.peaks.mz[condition],
            intensities=spectrum.peaks.intensities[condition]
        )

        return spectrum

    def _create_condition(self, spectrum: SpectrumType):
        raise NotImplementedError("Subclasses must implement _create_condition method.")