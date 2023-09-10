import logging
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")

class DeriveFromInchiSmileTemplate(BaseSpectrumFilter):
    def __init__(self, derive_to, derive_from, convert_function, log_message):
        self.derive_to = derive_to
        self.derive_from = derive_from
        self.convert_function = convert_function
        self.log_message = log_message

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        derive_to = spectrum.get(self.derive_to)
        derive_from = spectrum.get(self.derive_from)

        if self.is_valid(derive_to, derive_from):
            converted_value = self.convert_function(derive_from)
            if converted_value:
                converted_value = converted_value.rstrip()
                spectrum.set(self.derive_to, converted_value)
                logger.info(self.log_message % converted_value)
            else:
                logger.warning("Could not convert %s %s to %s.", self.derive_from, derive_from, self.derive_to)

        return spectrum

    def is_valid(self, input_value, output_value):
        raise NotImplementedError("Subclasses must implement this method")

