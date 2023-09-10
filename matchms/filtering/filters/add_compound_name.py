import logging
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")

class AddCompoundName(BaseSpectrumFilter):
    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        if spectrum.get("compound_name", None) is None:
            if isinstance(spectrum.get("name", None), str):
                spectrum.set("compound_name", spectrum.get("name"))
                return spectrum

            if isinstance(spectrum.get("title", None), str):
                spectrum.set("compound_name", spectrum.get("title"))
                return spectrum

            logger.info("No compound name found in metadata.")

        return spectrum
