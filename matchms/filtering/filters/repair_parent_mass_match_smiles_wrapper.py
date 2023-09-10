import logging
from typing import Optional
from matchms.filtering.metadata_processing.repair_adduct_based_on_smiles import \
    repair_adduct_based_on_smiles
from matchms.typing import SpectrumType
from ..metadata_processing.repair_parent_mass_is_mol_wt import repair_parent_mass_is_mol_wt
from ..metadata_processing.repair_precursor_is_parent_mass import repair_precursor_is_parent_mass
from ..metadata_processing.repair_smiles_of_salts import repair_smiles_of_salts
from ..metadata_processing.require_parent_mass_match_smiles import require_parent_mass_match_smiles
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


class RepairParentMassMatchSmilesWrapper(BaseSpectrumFilter):
    def __init__(self, mass_tolerance: float = 0.2):
        self.mass_tolerance = mass_tolerance

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        if require_parent_mass_match_smiles(spectrum, self.mass_tolerance) is not None:
            return spectrum

        spectrum = repair_smiles_of_salts(spectrum, self.mass_tolerance)
        if require_parent_mass_match_smiles(spectrum, self.mass_tolerance) is not None:
            return spectrum

        spectrum = repair_precursor_is_parent_mass(spectrum, self.mass_tolerance)
        if require_parent_mass_match_smiles(spectrum, self.mass_tolerance) is not None:
            return spectrum

        spectrum = repair_parent_mass_is_mol_wt(spectrum, self.mass_tolerance)
        if require_parent_mass_match_smiles(spectrum, self.mass_tolerance) is not None:
            return spectrum

        spectrum = repair_adduct_based_on_smiles(spectrum, self.mass_tolerance, accept_parent_mass_is_mol_wt=True)
        if require_parent_mass_match_smiles(spectrum, self.mass_tolerance) is not None:
            return spectrum
        return None
