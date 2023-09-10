from typing import Optional
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import \
    get_monoisotopic_neutral_mass
from matchms.typing import SpectrumType
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter


class RequireParentMassMatchSmiles(BaseSpectrumFilter):
    def __init__(self, mass_tolerance):
        self.mass_tolerance = mass_tolerance

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        # Check if parent mass matches the smiles mass
        parent_mass = spectrum.get("parent_mass")
        smiles = spectrum.get("smiles")
        smiles_mass = get_monoisotopic_neutral_mass(smiles)
        mass_difference = parent_mass - smiles_mass
        if abs(mass_difference) < self.mass_tolerance:
            return spectrum
        return None
