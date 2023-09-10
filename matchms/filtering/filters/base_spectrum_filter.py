from abc import ABC, abstractmethod
from matchms.typing import SpectrumType


class BaseSpectrumFilter(ABC):
    def process(self, spectrum_in: SpectrumType) -> SpectrumType:
        if spectrum_in is None:
            return None

        spectrum = spectrum_in.clone()

        spectrum = self.apply_filter(spectrum)

        return spectrum

    @abstractmethod
    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        """This method should be implemented by child classes."""
        raise NotImplementedError