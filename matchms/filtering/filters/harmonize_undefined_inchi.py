from matchms.filtering.filters.harmonize_undefined_template import HarmonizeUndefinedTemplate
from matchms.typing import SpectrumType


class HarmonizeUndefinedInchi(HarmonizeUndefinedTemplate):
    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        return super().apply_filter(spectrum, "inchi")