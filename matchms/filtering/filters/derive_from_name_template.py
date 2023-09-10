from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.typing import SpectrumType


class DeriveFromNameTemplate(BaseSpectrumFilter):
    def __init__(self, remove_from_name: bool, metadata_key: str):
        self.remove_from_name = remove_from_name
        self.metadata_key = metadata_key

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        if spectrum.get("compound_name", None) is not None:
            name = spectrum.get("compound_name")
        else:
            assert spectrum.get("name", None) in [None, ""], ("Found 'name' but not 'compound_name' in metadata",
                                                              "Apply 'add_compound_name' filter first.")
            return spectrum

        spectrum = self.derive(name, spectrum)

        return spectrum

    def derive(self, name, spectrum):
        raise NotImplementedError("Subclasses must implement the 'derive' method.")