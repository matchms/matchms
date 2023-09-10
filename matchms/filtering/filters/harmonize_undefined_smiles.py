from matchms.filtering.filters.harmonize_undefined_template import HarmonizeUndefinedTemplate
from matchms.typing import SpectrumType


class HarmonizeUndefinedSmiles(HarmonizeUndefinedTemplate):
    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        return super().apply_filter(spectrum, "smiles")