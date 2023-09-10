from typing import List
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.typing import SpectrumType


class HarmonizeUndefinedTemplate(BaseSpectrumFilter):
    def __init__(self, undefined: str = "", aliases: List[str] = None):
        self.undefined = undefined
        self.aliases = aliases or ["", "N/A", "NA", "n/a", "no data"]

    def apply_filter(self, spectrum: SpectrumType, key: str) -> SpectrumType:
        value = spectrum.get(key)
        if value is None:
            # spectrum does not have the specified key in its metadata
            spectrum.set(key, self.undefined)
            return spectrum

        if value in self.aliases:
            # harmonize aliases for undefined values
            spectrum.set(key, self.undefined)

        return spectrum