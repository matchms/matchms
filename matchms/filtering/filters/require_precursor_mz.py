import logging
from typing import Union
from matchms.typing import SpectrumType
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter


logger = logging.getLogger("matchms")


class RequirePrecursorMz(BaseSpectrumFilter):
    def __init__(self, minimum_accepted_mz: float = 10.0):
        self.minimum_accepted_mz = minimum_accepted_mz

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        precursor_mz = spectrum.get("precursor_mz", None)
        if precursor_mz is None:
            pepmass = spectrum.get("pepmass", None)
            assert pepmass is None or not isinstance(pepmass[0], (float, int)), \
                "Found 'pepmass' but no 'precursor_mz'. " \
                "Consider applying 'add_precursor_mz' filter first."
            return None

        assert isinstance(precursor_mz, (float, int)), \
            ("Expected 'precursor_mz' to be a scalar number.",
             "Consider applying 'add_precursor_mz' filter first.")
        if precursor_mz <= self.minimum_accepted_mz:
            logger.info("Spectrum without precursor_mz was set to None.")
            return None

        return spectrum
